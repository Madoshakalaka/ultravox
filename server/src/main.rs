//! this axum app is the backend wrapper for an LLM.
//! the core AppState stores sessions. The key of each session is u32. The value use tokio channels
//! to get output from the LLM.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use tokio::sync::mpsc;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
// use tower_http::cors::CorsLayer;

// Type alias for our session ID
type SessionId = u32;

// Channel types for LLM communication
type LlmSender = mpsc::Sender<String>;
type LlmReceiver = mpsc::Receiver<String>;

// Structure to hold active sessions
#[derive(Default)]
struct Sessions {
    sessions: HashMap<SessionId, LlmSender>,
}

// Our main application state
#[derive(Clone)]
struct AppState {
    sessions: Arc<RwLock<Sessions>>,
    next_id: Arc<AtomicU32>,
}

// Request and response structures
#[derive(Deserialize)]
struct PromptRequest {
    session_id: SessionId,
    prompt: String,
}

#[derive(Serialize)]
struct NewSessionResponse {
    session_id: SessionId,
}

impl AppState {
    fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(Sessions::default())),
            next_id: Arc::new(AtomicU32::new(1)),
        }
    }
}

// Handler to create a new session
async fn create_session(State(state): State<AppState>) -> Json<NewSessionResponse> {
    let session_id = state.next_id.fetch_add(1, Ordering::SeqCst);
    let (tx, _rx) = mpsc::channel(32);

    let mut sessions = state.sessions.write().await;
    sessions.sessions.insert(session_id, tx);

    Json(NewSessionResponse { session_id })
}

// Handler to send a prompt to the LLM
async fn send_prompt(
    State(state): State<AppState>,
    Json(request): Json<PromptRequest>,
) -> Result<Json<String>, String> {
    let sessions = state.sessions.read().await;

    if let Some(sender) = sessions.sessions.get(&request.session_id) {
        // Here you would typically send the prompt to your LLM processing logic
        // For now, we'll just echo back the prompt
        sender
            .send(request.prompt.clone())
            .await
            .map_err(|e| e.to_string())?;

        Ok(Json(format!(
            "Prompt received for session {}",
            request.session_id
        )))
    } else {
        Err(format!("Session {} not found", request.session_id))
    }
}

/// A minimal struct to represent an incoming "voice sample" or message set.
/// Adjust this to match your real data.
#[derive(Debug)]
struct VoiceSample {
    messages: Vec<(String, String)>, // e.g. [("role", "user"), ("content", "Hello")]
}

fn run_inference(
    inference_py: &'static CStr,
    mut prompt_rx: mpsc::Receiver<VoiceSample>,
) -> Result<mpsc::Receiver<String>, ()> {
    // Required if you are going to use Python from multiple threads:
    pyo3::prepare_freethreaded_python();

    // Create a channel that will carry VoiceSample messages for inference.
    let (tx, mut inference_rx) = mpsc::channel::<String>(16);

    tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let my_infer_module =
                PyModule::from_code(py, inference_py, c_str!("main.py"), c_str!("main"))?;

            let local_inference_class = my_infer_module.getattr("LocalInference")?;

            let model_py = py.None();
            let processor_py = py.None();
            let tokenizer_py = py.None();
            let device = "cpu";
            let dtype = "float32";
            let conversation_mode = false;

            let local_inference_instance = local_inference_class.call1((
                model_py,
                processor_py,
                tokenizer_py,
                device,
                dtype,
                conversation_mode,
            ))?;

            while let Some(sample) = prompt_rx.blocking_recv() {
                // Build a Python dict that replicates the `sample` shape.
                let sample_dict = PyDict::new(py);

                // Our Python code expects sample.messages to be a list of { "role": "...", "content": "..." }
                let mut py_messages = Vec::new();
                for (role, content) in &sample.messages {
                    let msg_dict = PyDict::new(py);
                    msg_dict.set_item("role", role)?;
                    msg_dict.set_item("content", content)?;
                    py_messages.push(msg_dict);
                }
                use pyo3::types::PyDictMethods;

                sample_dict.set_item("messages", py_messages)?;

                // Call local_inference_instance.infer(sample_dict)
                let inference_result =
                    local_inference_instance.call_method1("infer", (sample_dict,))?;

                // The returned object is a VoiceOutput with a .text attribute:
                let answer_text: String = inference_result.getattr("text")?.extract()?;
                tx.blocking_send(answer_text)
                    .expect("Failed to send inference result");
            }

            Ok::<(), PyErr>(())
        })
        .expect("Python GIL block failed");
    });

    Ok(inference_rx)
}

#[tokio::main]
async fn main() {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    let app_state = AppState::new();

    // Build our application with routes
    let app = Router::new()
        .route("/session/new", post(create_session))
        .route("/prompt", post(send_prompt))
        // .layer(CorsLayer::permissive())
        .with_state(app_state);

    // Run our application
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("Server running on http://127.0.0.1:3000");

    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[tokio::test]
    async fn test_run_inference() {
        let py_mockmain = c_str!(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../mock_ultravox/main.py"
        )));

        let (tx, rx) = mpsc::channel(32);

        // Start the inference process
        let mut inference_rx = run_inference(py_mockmain, rx).expect("Failed to start inference");

        // Send a test message
        let test_message = VoiceSample {
            messages: vec![
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), "Hello, world!".to_string()),
            ],
        };

        tx.send(test_message)
            .await
            .expect("Failed to send test message");

        // Receive the response
        let response = inference_rx
            .recv()
            .await
            .expect("Failed to receive inference response");

        assert_eq!(response, "Hello, world!");
    }
}
