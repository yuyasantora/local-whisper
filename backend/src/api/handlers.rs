use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Response,
};
use tracing::{debug, info};

use crate::domain::models::TranscriptionResponse;
use crate::domain::vad::{VadConfig, VadState, VadStatus};

// WebSocketへのアップグレードハンドラー
// HTTPリクエストを受け取り、WebSocketへアップグレードする
pub async fn ws_handler(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    let mut vad_state = VadState::new(VadConfig::default());

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Binary(bytes) => {
                let pcm_data = bytes_to_f32(&bytes);
                // 2. VAD判定
                let status = vad_state.process(&pcm_data);

                // 3. 結果に応じてクライアントへ通知 (動作確認用)
                match status {
                    VadStatus::SpeechStarted => {
                        info!("🎤 User STARTED talking");
                        let resp = TranscriptionResponse::Status("listening".to_string());
                        let json = serde_json::to_string(&resp).unwrap();
                        let _ = socket.send(Message::Text(json.into())).await;
                    }
                    VadStatus::SpeechEnded => {
                        info!("✅ User STOPPED talking (Would trigger Whisper here)");
                        let resp = TranscriptionResponse::Status("processing".to_string());
                        let json = serde_json::to_string(&resp).unwrap();
                        let _ = socket.send(Message::Text(json.into())).await;
                    }
                    VadStatus::Speaking => {
                        debug!("... speaking (energy detected) ...");
                    }
                    _ => {}
                }
            }
            Message::Close(_) => {
                info!("Client disconnected");
                return;
            }
            _ => {}
        }
    }
}

/// ヘルパー: LittleEndianバイト列をf32に変換
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let buf: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(buf)
        })
        .collect()
}
