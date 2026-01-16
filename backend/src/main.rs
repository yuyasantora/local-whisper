mod api;
mod domain;
// mod services;
// mod infra;

use std::net::SocketAddr;
use tracing::info;

#[tokio::main]
async fn main() {
    // ログ設定
    tracing_subscriber::fmt::init();

    // ルーティング
    let app = api::router::app();

    let addr = SocketAddr::from(([127, 0, 0, 1], 8000));
    info!("🚀 Server listening on ws://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
