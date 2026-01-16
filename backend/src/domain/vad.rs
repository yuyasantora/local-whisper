/// VADの設定パラメータ
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// 音声とみなすエネルギー閾値 (0.0 ~ 1.0)
    /// マイク環境によりますが、0.005 ~ 0.02 くらいが一般的
    pub energy_threshold: f32,

    /// 発話終了とみなすまでの無音チャンク数
    /// 例: 1チャンク30msだとして、20チャンク=600ms無音が続けば文の区切りとする
    pub silence_threshold_chunks: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            silence_threshold_chunks: 15, // 約0.5秒程度の無音でカット
        }
    }
}

/// VADの判定結果
#[derive(Debug, PartialEq, Eq)]
pub enum VadStatus {
    /// 完全に無音
    Silence,
    /// 発話が始まった瞬間
    SpeechStarted,
    /// 発話中（継続）
    Speaking,
    /// 発話中の短い無音（文の区切りではない）
    SilenceDuringSpeech,
    /// 発話が終了した（文の区切り）
    SpeechEnded,
}

/// VADの状態管理（ステートマシン）
pub struct VadState {
    config: VadConfig,
    is_speaking: bool,
    silence_counter: usize,
}

impl VadState {
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            is_speaking: false,
            silence_counter: 0,
        }
    }

    /// 音声チャンク(PCM)を受け取り、状態遷移を判定する
    pub fn process(&mut self, pcm: &[f32]) -> VadStatus {
        let energy = self.calculate_rms(pcm);

        if energy > self.config.energy_threshold {
            // 音声を検知した場合
            self.silence_counter = 0;

            if !self.is_speaking {
                self.is_speaking = true;
                return VadStatus::SpeechStarted;
            }
            return VadStatus::Speaking;
        } else {
            // 音声レベル以下の無音の場合
            if self.is_speaking {
                self.silence_counter += 1;

                if self.silence_counter > self.config.silence_threshold_chunks {
                    // 閾値を超えて無音が続いた -> 発話終了
                    self.is_speaking = false;
                    self.silence_counter = 0;
                    return VadStatus::SpeechEnded;
                }
                // まだ文の途中かもしれない（息継ぎなど）
                return VadStatus::SilenceDuringSpeech;
            }
        }

        VadStatus::Silence
    }

    /// RMS (二乗平均平方根) で音量を計算
    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }
}
