import { useState, useRef, useCallback, useEffect } from 'react';

// サーバーからのメッセージ型定義 (Backendのdomain/models.rsと合わせる)
type ServerMessage =
  | { type: 'status'; text: string }
  | { type: 'partial'; text: string }
  | { type: 'final'; text: string };

export const useRealtimeTranscription = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState<string>('disconnected');
  const [transcript, setTranscript] = useState<string>('');

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // 1. WebSocket接続
  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('✅ WS Connected');
      setIsConnected(true);
      setStatus('ready');
    };

    ws.onmessage = (event) => {
      try {
        const data: ServerMessage = JSON.parse(event.data);
        if (data.type === 'status') setStatus(data.text);
        if (data.type === 'partial') setTranscript(prev => prev + ' ' + data.text); // 仮実装
        if (data.type === 'final') setTranscript(prev => prev + '\n' + data.text);
      } catch (e) {
        console.error('JSON parse error:', e);
      }
    };

    ws.onclose = () => {
      console.log('❌ WS Disconnected');
      setIsConnected(false);
      setStatus('disconnected');
    };

    socketRef.current = ws;
  }, []);

  // 2. 録音開始
  const startRecording = useCallback(async () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      alert("サーバーに接続されていません");
      return;
    }

    try {
      // マイク取得
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // AudioContext作成 (サンプリングレートはブラウザ依存だが、一旦そのまま送る)
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      // public/audio-processor.js を読み込む
      await audioContext.audioWorklet.addModule('/audio-processor.js');

      // ノード設定
      const source = audioContext.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(audioContext, 'audio-processor');

      // Workletからのデータ受信 -> WebSocket送信
      worklet.port.onmessage = (event) => {
        const inputData = event.data as Float32Array;
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          // 生のFloat32Arrayをそのままバイナリとして送信
          socketRef.current.send(inputData.buffer);
        }
      };

      source.connect(worklet);
      worklet.connect(audioContext.destination); // これがないと動かないブラウザがある
      workletNodeRef.current = worklet;

      setIsRecording(true);
      setStatus('listening');

    } catch (err) {
      console.error('Microphone error:', err);
      alert('マイクの起動に失敗しました');
    }
  }, []);

  // 3. 録音停止
  const stopRecording = useCallback(() => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    audioContextRef.current?.close();
    socketRef.current?.close(); // 今回は録音停止＝切断にする（Aqua Voice風なら維持でもOK）

    setIsRecording(false);
    setIsConnected(false);
    setStatus('disconnected');
  }, []);

  // クリーンアップ
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    isConnected,
    isRecording,
    status,
    transcript,
    connect,
    startRecording,
    stopRecording
  };
};
