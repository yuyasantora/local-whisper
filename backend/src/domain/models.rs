use serde::Serialize;

#[derive(Serialize, Debug)]
pub enum TranscriptionResponse {
    // 推論途中の中間結果
    Partial(String),
    // 確定文字列
    Final(String),
    // ステータス通知
    Status(String),
}

// サービス層からAPI層へ処理結果を渡すためのEnum
pub enum ProcessingEvent {
    // 発話警視
    SpeechStart,
    // 推論確定
    TransciptComplete(String),
    // 途中の更新
    TranscriptPartial(String),
    // 何も起きなかった場合
    None,
}
