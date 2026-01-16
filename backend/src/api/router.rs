use super::handlers::ws_handler;
use axum::{routing::get, Router};

pub fn app() -> Router {
    Router::new().route("/ws", get(ws_handler))
}
