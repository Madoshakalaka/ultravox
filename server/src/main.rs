use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::LazyLock;

use axum::http::StatusCode;
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

type LlmSender = mpsc::Sender<VoiceSample>;
type LlmReceiver = mpsc::Receiver<String>;

// Structure to hold active sessions
#[derive(Default)]
struct Sessions {
    sessions: HashMap<SessionId, (LlmSender, Arc<tokio::sync::Mutex<LlmReceiver>>)>,
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

#[derive(Serialize, Deserialize)]
struct NewSessionResponse {
    session_id: SessionId,
    first_llm_ouput: String,
}

impl AppState {
    fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(Sessions::default())),
            next_id: Arc::new(AtomicU32::new(1)),
        }
    }
}

async fn create_session(
    State(state): State<AppState>,
    Json(query): Json<String>,
) -> Json<NewSessionResponse> {
    let session_id = state.next_id.fetch_add(1, Ordering::SeqCst);
    let (tx, rx) = mpsc::channel(32);
    let mut llm_output = run_inference(rx).expect("Failed to start inference");
    tx.send(VoiceSample {
        message: query.clone(),
        first: true,
    })
    .await
    .expect("Failed to send first message");

    let first_llm_ouput = llm_output
        .recv()
        .await
        .expect("Failed to receive inference response");
    {
        let mut sessions = state.sessions.write().await;
        sessions.sessions.insert(
            session_id,
            (tx, Arc::new(tokio::sync::Mutex::new(llm_output))),
        );
    };
    Json(NewSessionResponse {
        session_id,
        first_llm_ouput,
    })
}

async fn send_prompt(
    State(state): State<AppState>,
    Json(request): Json<PromptRequest>,
) -> Result<Json<String>, (StatusCode, String)> {
    let recv = {
        let sessions = state.sessions.read().await;
        sessions
            .sessions
            .get(&request.session_id)
            .ok_or((StatusCode::NOT_FOUND, "Session not found".to_string()))?
            .1
            .clone()
    };

    let mut recv = recv.try_lock().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            "Session already in use".to_string(),
        )
    })?;

    {
        let sessions = state.sessions.read().await;
        let (tx, _) = sessions
            .sessions
            .get(&request.session_id)
            .ok_or((StatusCode::NOT_FOUND, "Session not found".to_string()))?;
        tx.send(VoiceSample {
            message: request.prompt.clone(),
            first: false,
        })
        .await
        .expect("Failed to send message");
    };

    let response = recv.recv().await.ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Failed to receive response".to_string(),
    ))?;

    Ok(Json(response))
}

#[derive(Debug)]
struct VoiceSample {
    message: String,
    first: bool,
}

static ULTRAVOX_SITE_PACKAGES: LazyLock<String> = LazyLock::new(|| {
    let output = std::process::Command::new("poetry")
        .arg("env")
        .arg("info")
        .arg("--path")
        .output()
        .expect("Failed to run poetry command");

    let path = std::str::from_utf8(&output.stdout).expect("Failed to parse output");
    let path = path.trim_end();
    let site_packages = format!("{}/lib/python3.10/site-packages", path);
    site_packages
});

fn run_inference(mut prompt_rx: mpsc::Receiver<VoiceSample>) -> Result<mpsc::Receiver<String>, ()> {
    // Required if you are going to use Python from multiple threads:
    pyo3::prepare_freethreaded_python();

    // Create a channel that will carry VoiceSample messages for inference.
    let (tx, inference_rx) = mpsc::channel::<String>(16);

    tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            // path.call_method1("append", (ULTRAVOX_SITE_PACKAGES.as_str(),))?;
            path.call_method1("insert", (0, ULTRAVOX_SITE_PACKAGES.as_str()))?;
            path.call_method1("insert", (0, "/home/maa/Projects/ultravox"))?;
            println!("Path: {:?}", path);

            // print!(env!("PYO3_PYTHON"));
            // print sys.executable
            let executable = sys.getattr("executable")?;
            println!("Executable: {:?}", executable);

            let my_infer_module = PyModule::import(py, "ultravox.inference.ultravox_infer")?;

            let data_sample_module = PyModule::import(py, "ultravox.data.data_sample")?;

            let inference_class = my_infer_module.getattr("UltravoxInference")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("conversation_mode", true)?;
            let inference_instance =
                inference_class.call(("fixie-ai/ultravox-v0_4_1-llama-3_1-8b",), Some(&kwargs))?;

            while let Some(VoiceSample { message, first }) = prompt_rx.blocking_recv() {
                // Create message dictionaries first
                let mut py_messages = Vec::new();

                if first {
                    // prepend a system message
                    let system_msg_dict = PyDict::new(py);
                    system_msg_dict.set_item("role", "system")?;
                    system_msg_dict.set_item("content", include_str!("system_prompt.txt"))?;
                    py_messages.push(system_msg_dict);
                }
                let user_msg_dict = PyDict::new(py);
                user_msg_dict.set_item("role", "user")?;
                user_msg_dict.set_item("content", message)?;
                py_messages.push(user_msg_dict);

                // Get the Sample class from the data_sample module
                let sample_class = data_sample_module.getattr("VoiceSample")?;

                // Create a new Sample instance with messages
                let sample_instance = sample_class.call1((py_messages,))?;

                // Call local_inference_instance.infer(sample_dict)
                let inference_result =
                    inference_instance.call_method1("infer", (sample_instance,))?;

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

async fn run(addr: String) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app_state = AppState::new();

    // Build our application with routes
    let app = Router::new()
        .route("/session", post(create_session))
        .route("/prompt", post(send_prompt))
        // .layer(CorsLayer::permissive())
        .with_state(app_state);

    // Run our application
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Server running on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

#[tokio::main]
async fn main() {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    if let Err(e) = run("127.0.0.1:3000".to_string()).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use std::net::TcpListener;

    fn find_available_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind random port");
        listener.local_addr().unwrap().port()
    }

    #[tokio::test]
    async fn create_session() {
        // Find an available port
        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        // Start the server in the background
        let _server = tokio::spawn(run(addr.clone()));

        // Give the server a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Create the client and send the request
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{}/session", addr))
            .json(&"Today my topic is Global Warming".to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(response.status(), StatusCode::OK);

        let body: NewSessionResponse = response.json().await.expect("Failed to parse response");
        assert!(body.session_id > 0);
        assert_eq!(body.first_llm_ouput, "Add Node: Global Warming");
    }
}
