// @ts-check

class AudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channelData = input[0];
      // float32Araryとして型推論される
      this.port.postMessage(channelData);
    }
    return true;
  }
}


globalThis.registerProcessor('audio-processor', AudioProcessor);
