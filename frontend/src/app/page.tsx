'use client'

import { useRealtimeTranscription } from '../hooks/useRealtimeTranscription'

export default function Home() {
  const {
    isConnected,
    isRecording,
    status,
    transcript,
    connect,
    startRecording,
    stopRecording,
  } = useRealtimeTranscription()


  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-8 text-blue-400">Local Aqua Voice</h1>

      <div className="flex gap-4 mb-8">
        {!isConnected ? (
          <button
            onClick={connect}
            className="px-6 py-3 rounded-full bg-blue-600 hover:bg-blue-700 font-semibold transition-all"
          >
            1. サーバー接続
          </button>
        ) : (
          <div className="flex gap-4">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="px-6 py-3 rounded-full bg-green-600 hover:bg-green-700 font-semibold transition-all"
              >
                2. マイクON (会話開始)
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="px-6 py-3 rounded-full bg-red-600 hover:bg-red-700 font-semibold transition-all animate-pulse"
              >
                ■ 停止
              </button>
            )}
          </div>
        )}
      </div>

      {/* ステータス表示 */}
      <div className="mb-6 px-4 py-2 rounded bg-gray-800 border border-gray-700">
        Status: <span className="font-mono text-yellow-400">{status}</span>
      </div>

      {/* 文字起こし結果エリア */}
      <div className="w-full max-w-2xl h-96 p-6 rounded-xl bg-gray-800 border border-gray-700 overflow-y-auto whitespace-pre-wrap font-sans text-lg leading-relaxed shadow-inner">
        {transcript || <span className="text-gray-500 italic">ここに文字起こし結果が表示されます...</span>}
      </div>
    </main>
  );
}
