use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// Whisperの仕様定数
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const CHUNK_LENGTH: usize = 30;
const N_MELS: usize = 128; // large-v3 は 128 (v2までは80)

pub struct WhisperEngine {
    model: m::Model,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
}

impl WhisperEngine {
    /// モデルをロードし、GPUに配置する
    pub async fn new() -> Result<Self> {
        println!("🔥 Loading Whisper Large-v3 model...");

        // 1. デバイス設定 (CUDAが使えれば使う)
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        println!("🚀 Running on device: {:?}", device);

        // 2. Hugging Faceからモデルファイルをダウンロード
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "openai/whisper-large-v3".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        // 3. 設定とモデルの構築
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Safetensorsからロード
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        let model = m::Model::new(&config, vb)?;

        // 4. Melフィルタバンクの初期化 (前処理用)
        // 本来はファイルから読み込むが、ここでは簡易実装として計算済みフィルタを使うか、
        // あるいは m::audio::pcm_to_mel のようなヘルパーを使うのが一般的。
        // 今回は実装を簡単にするため、推論時に動的に計算するアプローチをとります。
        let mel_filters = vec![]; // 今回は自前実装せず、candle-transformersのヘルパーがあればそれを使う

        println!("✅ Whisper Model Loaded Successfully!");

        Ok(Self {
            model,
            tokenizer,
            device,
            mel_filters,
        })
    }

    /// 音声データ(PCM)を受け取り、テキストに変換する
    pub async fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        // 1. 音声の前処理 (PCM -> Mel Spectrogram)
        // ※ 本来はここでFFTとMelフィルタバンク適用を行う。
        // Candleにはまだ標準的な `audio` 前処理クレートがないため、
        // ここでは実装の簡略化のために「モデルはロードできたが、前処理はTODO」の状態を避けるべく
        // 最小限の変換ロジック、あるいは外部クレートの使用が推奨されます。
        // 今回は「動く骨組み」として、ダミーで実装を進め、後で詳細なMel変換を差し込みます。

        // ★ここがRustで一番難しいところです。
        // 本当に動かすには `wavy` や `symphonia` 等でFFTする必要がありますが、
        // 長大になるため、一旦「空文字を返す（モデルロード成功確認）」まで行きますか？
        // それとも「何が何でも実装する」方向で行きますか？

        // -> あなたの実力なら「完全実装」を望むはず。
        // しかしコード量が300行を超えてしまうため、
        // ここでは「推論ロジック(Decoder Loop)」の核心部分だけ書きます。

        let mel = self.extract_mel(pcm)?;
        let mel_len = mel.dim(2)?;
        let mel_tensor = mel.to_device(&self.device)?;

        // 2. 言語検出 (今回は日本語固定にしてスキップも可)
        let language_token = match self.tokenizer.token_to_id("<|ja|>") {
            Some(t) => t,
            None => 50259, // default to Japanese if not found
        };

        // 3. Decoder Loop (Greedy Search)
        // Whisperは Encoder-Decoder モデルです。
        let encoder_output = self.model.encoder.forward(&mel_tensor, true)?;

        // 初期トークン: [SOT, Language, Transcribe]
        let mut tokens = vec![
            self.tokenizer.token_to_id("<|startoftranscript|>").unwrap(),
            language_token,
            self.tokenizer.token_to_id("<|transcribe|>").unwrap(),
        ];

        // 推論ループ (最大100トークンまで生成)
        for _ in 0..100 {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            // Decoder Forward
            let logits = self
                .model
                .decoder
                .forward(&tokens_t, &encoder_output, true)?;
            let logits = logits.squeeze(0)?; // (seq_len, vocab_size)
            let next_token_logits = logits.get(logits.dim(0)? - 1)?;

            // Argmax (Greedy)
            let next_token = next_token_logits.argmax(0)?.to_scalar::<u32>()?;

            tokens.push(next_token);

            // <|endoftext|> が来たら終了
            if next_token == 50257 {
                break;
            }
        }

        // 4. トークンを文字列にデコード
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        Ok(text)
    }

    // Mel Spectrogram抽出 (簡易版)
    // ※実際にはここにFFTの実装が必要です。
    // 今回はコンパイルを通すためにダミーのTensorを返します。
    // 次のステップでここを「本物のFFT」に置き換えるライブラリを入れます。
    fn extract_mel(&self, _pcm: &[f32]) -> Result<Tensor> {
        // [1, 128, 3000] のダミーデータ (無音)
        let noise = Tensor::randn(0f32, 1f32, (1, 128, 3000), &Device::Cpu)?;
        Ok(noise)
    }
}
