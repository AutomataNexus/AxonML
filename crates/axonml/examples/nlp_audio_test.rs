//! NLP and Audio Test Example
//!
//! Tests text tokenization, vocabulary, and audio transforms.

use axonml::prelude::*;

fn main() {
    println!("=== Axonml ML Framework - NLP & Audio Test ===\n");

    // === TEXT PROCESSING ===
    println!("--- TEXT PROCESSING ---\n");

    // 1. Tokenization
    println!("1. Testing Tokenizers...");
    let text = "Hello, world! This is Axonml ML framework.";

    let ws_tokenizer = WhitespaceTokenizer::new();
    let ws_tokens = ws_tokenizer.tokenize(text);
    println!("   Whitespace: {ws_tokens:?}");

    let char_tokenizer = CharTokenizer::new();
    let char_tokens = char_tokenizer.tokenize("Hello");
    println!("   Character: {char_tokens:?}");

    // 2. Vocabulary
    println!("\n2. Building Vocabulary...");
    let corpus = "the quick brown fox jumps over the lazy dog";
    let vocab = Vocab::from_text(corpus, 1);
    println!("   Vocabulary size: {}", vocab.len());
    println!("   'the' index: {:?}", vocab.token_to_index("the"));
    println!("   'quick' index: {:?}", vocab.token_to_index("quick"));

    // 3. Encode/Decode
    println!("\n3. Encoding text...");
    let encoded = ws_tokenizer.encode("the quick fox", &vocab);
    println!("   'the quick fox' -> {encoded:?}");

    // 4. BPE Tokenizer
    println!("\n4. Training BPE Tokenizer...");
    let mut bpe = BasicBPETokenizer::new();
    bpe.train("low lower lowest newer newest", 10);
    let bpe_tokens = bpe.tokenize("lower");
    println!("   BPE vocab size: {}", bpe.get_vocab().len());
    println!("   'lower' tokens: {bpe_tokens:?}");

    // 5. Text Dataset
    println!("\n5. Creating Sentiment Dataset...");
    let sentiment_dataset = SyntheticSentimentDataset::small();
    println!("   Dataset size: {}", sentiment_dataset.len());

    let (text_tensor, label) = sentiment_dataset.get(0).unwrap();
    println!("   Sample text shape: {:?}", text_tensor.shape());
    println!("   Sample label shape: {:?}", label.shape());

    // === AUDIO PROCESSING ===
    println!("\n--- AUDIO PROCESSING ---\n");

    // 6. Audio Transforms
    println!("6. Testing Audio Transforms...");

    // Create a simple sine wave
    use std::f32::consts::PI;
    let sample_rate = 16000;
    let duration = 0.5;
    let frequency = 440.0; // A4 note

    let n_samples = (sample_rate as f32 * duration) as usize;
    let audio_data: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate as f32).sin())
        .collect();
    let audio = Tensor::from_vec(audio_data, &[n_samples]).unwrap();

    println!("   Input audio: {n_samples} samples at {sample_rate}Hz");

    // Resample
    let resample = Resample::new(16000, 8000);
    let resampled = resample.apply(&audio);
    println!(
        "   Resampled: {} -> {} samples",
        n_samples,
        resampled.shape()[0]
    );

    // Mel Spectrogram
    let mel = MelSpectrogram::with_params(16000, 512, 256, 40);
    let spectrogram = mel.apply(&audio);
    println!("   Mel Spectrogram shape: {:?}", spectrogram.shape());

    // MFCC
    let mfcc = MFCC::new(16000, 13);
    let mfcc_features = mfcc.apply(&audio);
    println!("   MFCC shape: {:?}", mfcc_features.shape());

    // Normalize
    let normalize = NormalizeAudio::new();
    let normalized = normalize.apply(&audio);
    let max_val = normalized
        .to_vec()
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    println!("   Normalized max amplitude: {max_val:.4}");

    // Add Noise
    let add_noise = AddNoise::new(20.0);
    let noisy = add_noise.apply(&audio);
    println!("   Noisy audio shape: {:?}", noisy.shape());

    // 7. Audio Dataset
    println!("\n7. Creating Audio Datasets...");

    let command_dataset = SyntheticCommandDataset::small();
    println!(
        "   Command dataset: {} samples, {} classes",
        command_dataset.len(),
        command_dataset.num_classes()
    );

    let music_dataset = SyntheticMusicDataset::small();
    println!(
        "   Music dataset: {} samples, {} genres",
        music_dataset.len(),
        music_dataset.num_genres()
    );

    let (wave, label) = command_dataset.get(0).unwrap();
    println!("   Sample waveform shape: {:?}", wave.shape());
    println!("   Sample label shape: {:?}", label.shape());

    println!("\n=== All Tests Passed! ===");
}
