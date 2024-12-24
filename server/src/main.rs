use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use tokio::sync::RwLock;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

type SessionId = u32;

// ---------- Conversation storage models ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Default)]
struct Sessions {
    // For each session, we store a vector of messages
    // (system, user, assistant, etc.)
    history: HashMap<SessionId, Vec<ChatMessage>>,
}

// ---------- AppState holds the single UltravoxInference instance & sessions ----------

#[derive(Clone)]
struct AppState {
    // The single Python inference object
    inference: Arc<Py<PyAny>>,

    // The conversation history for all sessions
    sessions: Arc<RwLock<Sessions>>,

    // ID counter for new sessions
    next_id: Arc<AtomicU32>,
}

// We will fill this in `main` after we create the Py<PyAny> from Python
impl AppState {
    fn new(inference: Py<PyAny>) -> Self {
        Self {
            inference: Arc::new(inference),
            sessions: Arc::new(RwLock::new(Sessions::default())),
            next_id: Arc::new(AtomicU32::new(1)),
        }
    }
}

// ---------- Request/Response Structs ----------

#[derive(Deserialize)]
struct PromptRequest {
    session_id: SessionId,
    prompt: String,
}

#[derive(Serialize, Deserialize)]
struct NewSessionResponse {
    session_id: SessionId,
    first_llm_output: String,
}

// ---------- HTTP Handlers ----------

/// Create a new session:
/// 1. Generate a new session ID.
/// 2. Initialize that session’s conversation in the map.
/// 3. Optionally add a system or welcome message.
/// 4. Return the first assistant output if needed.
async fn create_session(
    State(state): State<AppState>,
    Json(system_message): Json<String>,
) -> Json<NewSessionResponse> {
    // Generate session ID
    let session_id = state.next_id.fetch_add(1, Ordering::SeqCst);

    // We store a system message in the conversation so that we can
    // show the model context. Or you might skip it if you do not need a system prompt.
    {
        let mut sessions = state.sessions.write().await;
        sessions.history.insert(
            session_id,
            vec![ChatMessage {
                role: "system".to_string(),
                content: system_message.clone(),
            }],
        );
    }

    // If you want an immediate, “first” model response, you can do:
    let first_llm_output = match run_inference(&state, session_id).await {
        Ok(text) => text,
        Err(e) => {
            eprintln!("Error in first inference: {}", e);
            "Error in first inference".to_string()
        }
    };

    Json(NewSessionResponse {
        session_id,
        first_llm_output,
    })
}

/// Send a user prompt and get the model’s reply.
/// 1. Append a new user message to the session’s conversation.
/// 2. Call inference with the entire conversation so far.
/// 3. Return the last assistant message to the client.
async fn send_prompt(
    State(state): State<AppState>,
    Json(request): Json<PromptRequest>,
) -> Result<Json<String>, (StatusCode, String)> {
    let PromptRequest { session_id, prompt } = request;

    // 1. Get the session’s conversation from the map
    {
        let mut sessions = state.sessions.write().await;

        // Return an error if this session does not exist
        let conv = sessions.history.get_mut(&session_id);
        if conv.is_none() {
            return Err((StatusCode::NOT_FOUND, "Session not found".to_string()));
        }

        // 2. Push the new user message into the history
        let conv = conv.unwrap();
        conv.push(ChatMessage {
            role: "user".to_string(),
            content: prompt,
        });
    }

    // 3. Call inference
    let assistant_reply = run_inference(&state, session_id)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err))?;

    Ok(Json(assistant_reply))
}

// ---------- The actual inference call ----------

/// Takes the entire conversation for `session_id` and sends it to Python.
/// Returns the assistant's text reply.  
///
/// This is where we gather all messages and build the Python side `VoiceSample`.
async fn run_inference(state: &AppState, session_id: SessionId) -> Result<String, String> {
    // 1. Clone the entire conversation
    let conv_snapshot = {
        let sessions = state.sessions.read().await;
        sessions
            .history
            .get(&session_id)
            .cloned()
            .ok_or_else(|| "Session not found".to_string())?
    };

    // 2. Invoke Python
    //    We'll create the list of message dicts and feed them into the shared UltravoxInference.
    Python::with_gil(|py| {
        let inference_instance = state.inference.as_ref();

        // Import the data_sample module so we can build a VoiceSample
        let data_sample_module = PyModule::import(py, "ultravox.data.data_sample")
            .map_err(|e| format!("Failed to import data_sample: {:?}", e))?;

        let sample_class = data_sample_module
            .getattr("VoiceSample")
            .map_err(|e| format!("Failed to get VoiceSample class: {:?}", e))?;

        // Build a Python list of message dicts
        let mut py_messages = Vec::new();
        for msg in conv_snapshot.iter() {
            let dict = PyDict::new(py);
            dict.set_item("role", &msg.role)
                .map_err(|e| format!("Failed to set role: {:?}", e))?;
            dict.set_item("content", &msg.content)
                .map_err(|e| format!("Failed to set content: {:?}", e))?;
            py_messages.push(dict);
        }

        // Create a new VoiceSample instance from that list
        let sample_instance = sample_class
            .call1((py_messages,))
            .map_err(|e| format!("Failed to create sample instance: {:?}", e))?;

        // inference_instance is your single shared `UltravoxInference`
        let inference_result = inference_instance
            .call_method1(py, "infer", (sample_instance,))
            .map_err(|e| format!("Call to infer(...) failed: {:?}", e))?;

        // Extract the returned string
        let answer_text: String = inference_result
            .getattr(py, "text")
            .map_err(|e| format!("No 'text' attr: {:?}", e))?
            .extract(py)
            .map_err(|e| format!("Failed to extract text: {:?}", e))?;

        // 3. Append the assistant’s message to the conversation in Rust
        {
            let mut sessions = state.sessions.blocking_write();
            if let Some(conv) = sessions.history.get_mut(&session_id) {
                conv.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: answer_text.clone(),
                });
            }
        }

        // Return the string back out
        Ok(answer_text)
    })
}

// ---------- The single shared Inference initialization at app startup ----------

static ULTRAVOX_SITE_PACKAGES: LazyLock<String> = LazyLock::new(|| {
    let output = std::process::Command::new("poetry")
        .arg("env")
        .arg("info")
        .arg("--path")
        .output()
        .expect("Failed to run poetry command");

    let path = std::str::from_utf8(&output.stdout).expect("Failed to parse output");
    let path = path.trim_end();
    format!("{}/lib/python3.10/site-packages", path)
});

/// Create the single Python `UltravoxInference` instance once.
fn init_inference() -> Py<PyAny> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<_> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, ULTRAVOX_SITE_PACKAGES.as_str()))?;
        path.call_method1("insert", (0, "/home/maa/Projects/ultravox"))?;

        let my_infer_module = PyModule::import(py, "ultravox.inference.ultravox_infer")?;
        let inference_class = my_infer_module.getattr("UltravoxInference")?;

        let inference_instance =
                    inference_class.call_method1("__init__", ("fixie-ai/ultravox-v0_4_1-llama-3_1-8b",))?;

        Ok(inference_instance.into())
    })
    .expect("Failed to initialize global UltravoxInference")
}

// ---------- Main server ----------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Create our single shared Python inference instance
    let py_inference = init_inference();

    // Build the app state from that instance
    let app_state = AppState::new(py_inference);

    // Build the router
    let app = Router::new()
        .route("/session", post(create_session))
        .route("/prompt", post(send_prompt))
        .with_state(app_state);

    // Start the server
    let addr = "127.0.0.1:3000";
    println!("Server running on http://{}", addr);

    // Run our application
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("Server running on http://{}", addr);

    axum::serve(listener, app).await.unwrap()
}


