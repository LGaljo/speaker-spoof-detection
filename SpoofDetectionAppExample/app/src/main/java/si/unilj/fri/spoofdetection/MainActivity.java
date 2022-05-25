package si.unilj.fri.spoofdetection;

import static java.nio.ByteOrder.BIG_ENDIAN;
import static java.nio.ByteOrder.LITTLE_ENDIAN;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.WindowManager;

import androidx.annotation.LongDef;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.os.HandlerCompat;

import com.paramsen.noise.Noise;

import org.tensorflow.lite.support.audio.TensorAudio;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.audio.classifier.AudioClassifier;
import org.tensorflow.lite.task.audio.classifier.Classifications;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import si.unilj.fri.spoofdetection.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    private ActivityMainBinding binding;

    ProbabilitiesAdapter probabilitiesAdapter;
    AudioClassifier audioClassifier = null;
    AudioRecord audioRecord = null;
    long classificationInterval = 500L;

    private Handler handler; // background thread handler to run classification
    private Handler handler_rec; // background thread handler to run recording

    private final int REQUEST_RECORD_AUDIO = 1337;
    private final String MODEL_FILE = "spoof_detection_metadata.tflite";
    //    private final String MODEL_FILE = "yamnet.tflite";
    private final float MINIMUM_DISPLAY_THRESHOLD = 0.01f;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        this.probabilitiesAdapter = new ProbabilitiesAdapter();
        binding.recyclerView.setHasFixedSize(false);
        binding.recyclerView.setAdapter(this.probabilitiesAdapter);

        this.keepScreenOn(binding.inputSwitch.isChecked());
        binding.inputSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) startAudioClassification();
            else stopAudioClassification();
            keepScreenOn(isChecked);
        });

        // Slider which control how often the classification task should run
        binding.classificationIntervalSlider.setValue((float) classificationInterval);
        binding.classificationIntervalSlider.setLabelFormatter(value -> "${value.toInt()} ms");
        binding.classificationIntervalSlider.addOnChangeListener((slider, value, fromUser) -> {
            classificationInterval = (long) value;
            stopAudioClassification();
            startAudioClassification();
        });

        // Create a handler to run classification in a background thread
        HandlerThread handlerThread = new HandlerThread("backgroundThread");
        handlerThread.start();
        HandlerThread handlerThread_rec = new HandlerThread("recordingThread");
        handlerThread_rec.start();
        handler = HandlerCompat.createAsync(handlerThread.getLooper());
        handler_rec = HandlerCompat.createAsync(handlerThread_rec.getLooper());

        // Request microphone permission and start running classification
        requestMicrophonePermission();

        ArrayList<Float> tensor = new ArrayList<>();

//        try {
//            float[] wavfile = readingAudioFile("genuine_Luka_s20fe_1.wav");
//            Log.d(TAG, "Length " + wavfile.length);
//            int width = 1024;
//            float[] buffer = new float[width];
//            for (int i = 0; i < 61 * 512; i += 512) {
//                System.arraycopy(wavfile, i, buffer, 0, width);
//                Float[] values = FftGen.runFft(buffer, 1024);
//                Collections.addAll(tensor, values);
//            }
//            Log.d(TAG, "Length " + tensor.size());
//            AudioClassifier.AudioClassifierOptions options = AudioClassifier.AudioClassifierOptions.builder().build();
//            final AudioClassifier classifier = AudioClassifier.createFromFileAndOptions(this, MODEL_FILE, options);
//
//            float[] floats = new float[tensor.size()];
//            for (int i = 0; i < tensor.size(); i++) {
//                floats[i] = (float) tensor.get(i);
//            }
//            TensorAudio audioTensor = classifier.createInputTensorAudio();
//            audioTensor.load(floats);
//            List<Classifications> output = classifier.classify(audioTensor);
//            Log.d(TAG, output.toArray().toString());
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

//        try {
//            AssetManager am = getAssets();
//            InputStream file = am.open("encoded.txt");
//            BufferedReader br = new BufferedReader(new InputStreamReader(file));
//            String line;
//
//            int i = 0;
//            while ((line = br.readLine()) != null) {
//                samples[i] = Float.parseFloat(line);
//                i++;
//            }
//            br.close();
//
//        } catch (IOException ex) {
//            ex.printStackTrace();
//        }
    }

    @SuppressLint("MissingPermission")
    private void startAudioClassification() {
        // If the audio classifier is initialized and running, do nothing.
        if (audioClassifier != null) return;

        // Initialize the audio classifier
        AudioClassifier.AudioClassifierOptions options = AudioClassifier.AudioClassifierOptions.builder().build();
        try {
            final AudioClassifier classifier = AudioClassifier.createFromFileAndOptions(this, MODEL_FILE, options);

            final TensorAudio audioTensor = classifier.createInputTensorAudio();

            // Initialize the audio recorder
            AudioRecord record = classifier.createAudioRecord();
            record.startRecording();

//            int buf_len = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT);
//            AudioRecord record = new AudioRecord(
//                    MediaRecorder.AudioSource.UNPROCESSED,
//                    16000,
//                    AudioFormat.CHANNEL_IN_MONO,
//                    AudioFormat.ENCODING_PCM_FLOAT,
//                    4096
//            );
//            record.startRecording();

            ArrayList<Float> tensor = new ArrayList<>();
            Runnable run_recording = new Runnable() {
                @Override
                public void run() {
                    float[] audioBuffer = new float[1024];

                    record.read(audioBuffer, 0, 1024, AudioRecord.READ_NON_BLOCKING);

//                    Log.d(TAG, "state: " + audioRecord.getState());
//                    Log.d(TAG, "state: " + audioRecord.getRecordingState());
                    Float[] values = runFft(audioBuffer, 1024);
                    Collections.addAll(tensor, values);

                    float[] floats = new float[tensor.size()];
                    if (tensor.size() == 31293) {
                        for (int i = 0; i < tensor.size(); i++) {
                            floats[i] = (float) tensor.get(i);
                        }
                        tensor.clear();
                        audioTensor.load(floats);
                        Log.d(TAG, "Send new audioTensor to classifier");
                        handler_rec.postDelayed(this, classificationInterval);
                        return;
                    }

                    handler_rec.post(this);
                }
            };

            // Define the classification runnable
            Runnable run = new Runnable() {
                @Override
                public void run() {
                    long startTime = System.currentTimeMillis();

                    // Load the latest audio sample
//                    audioTensor.load(audioBuffer);

                    List<Classifications> output = classifier.classify(audioTensor);

                    // Filter out results above a certain threshold, and sort them descending
                    //        val filteredModelOutput = output[0].categories.filter {
                    //          it.score > MINIMUM_DISPLAY_THRESHOLD
                    //        }.sortedBy {
                    //          -it.score
                    //        }

                    ArrayList<Category> filteredModelOutput = new ArrayList<>(output.get(0).getCategories());

                    long finishTime = System.currentTimeMillis();

                    Log.d(TAG, String.format("Latency = %dms", finishTime - startTime));
                    Log.d(TAG, String.format("Score 0 = %f", filteredModelOutput.get(0).getScore()));
                    Log.d(TAG, String.format("Score 1 = %f", filteredModelOutput.get(1).getScore()));

                    // Updating the UI
                    runOnUiThread(() -> {
                        probabilitiesAdapter.categoryList = filteredModelOutput;
                        probabilitiesAdapter.notifyDataSetChanged();
                    });

                    // Rerun the classification after a certain interval
                    handler.postDelayed(this, classificationInterval);
                }
            };

            // Save the instances we just created for use later
            audioClassifier = classifier;
            audioRecord = record;

            // Start the classification process
            handler.post(run);
            handler_rec.post(run_recording);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void stopAudioClassification() {
        handler.removeCallbacksAndMessages(null);
        handler_rec.removeCallbacksAndMessages(null);
        audioRecord.stop();
//        audioRecord = null;
        audioClassifier = null;
    }

    @Override
    public void onTopResumedActivityChanged(boolean isTopResumedActivity) {
        // Handles "top" resumed event on multi-window environment
        if (isTopResumedActivity && isRecordAudioPermissionGranted()) {
            startAudioClassification();
        } else {
            stopAudioClassification();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Audio permission granted :)");
                startAudioClassification();
            } else {
                Log.e(TAG, "Audio permission not granted :(");
            }
        }
    }


    @RequiresApi(Build.VERSION_CODES.M)
    private void requestMicrophonePermission() {
        if (isRecordAudioPermissionGranted()) {
            startAudioClassification();
        } else {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    private boolean isRecordAudioPermissionGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED;
    }

    private void keepScreenOn(boolean enable) {
        if (enable) {
            this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            this.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    int[] type = {0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1};
    int[] numberOfBytes = {4, 4, 4, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4};
    int chunkSize, subChunk1Size, sampleRate, byteRate, subChunk2Size = 1, bytePerSample;
    short audioFormat, numChannels, blockAlign, bitsPerSample = 16;
    String chunkID, format, subChunk1ID, subChunk2ID;

    public float[] readingAudioFile(String audioFile) throws IOException {
        try {
//            AssetManager am = getAssets();
//            File file = new File(audioFile);
//            AssetFileDescriptor afd = am.openFd(audioFile);
//            int length = (int) file.length();
            //System.out.println(length);
//            InputStream fileInputstream = new FileInputStream(afd.getFileDescriptor());
            InputStream fileInputstream = getAssets().open(audioFile);
            int size = fileInputstream.available();

            ByteBuffer byteBuffer;
            for (int i = 0; i < numberOfBytes.length; i++) {
                byte[] byteArray = new byte[numberOfBytes[i]];
                int r = fileInputstream.read(byteArray, 0, numberOfBytes[i]);
                byteBuffer = ByteArrayToNumber(byteArray, numberOfBytes[i], type[i]);
                if (i == 0) {
                    chunkID = new String(byteArray);
                    System.out.println(chunkID);
                }
                if (i == 1) {
                    chunkSize = byteBuffer.getInt();
                    System.out.println(chunkSize);
                }
                if (i == 2) {
                    format = new String(byteArray);
                    System.out.println(format);
                }
                if (i == 3) {
                    subChunk1ID = new String(byteArray);
                    System.out.println(subChunk1ID);
                }
                if (i == 4) {
                    subChunk1Size = byteBuffer.getInt();
                    System.out.println(subChunk1Size);
                }
                if (i == 5) {
                    audioFormat = byteBuffer.getShort();
                    System.out.println(audioFormat);
                }
                if (i == 6) {
                    numChannels = byteBuffer.getShort();
                    System.out.println(numChannels);
                }
                if (i == 7) {
                    sampleRate = byteBuffer.getInt();
                    System.out.println(sampleRate);
                }
                if (i == 8) {
                    byteRate = byteBuffer.getInt();
                    System.out.println(byteRate);
                }
                if (i == 9) {
                    blockAlign = byteBuffer.getShort();
                    System.out.println(blockAlign);
                }
                if (i == 10) {
                    bitsPerSample = byteBuffer.getShort();
                    System.out.println(bitsPerSample);
                }
                if (i == 11) {
                    subChunk2ID = new String(byteArray);
                    if (subChunk2ID.compareTo("data") == 0) {
                        continue;
                    } else if (subChunk2ID.compareTo("LIST") == 0) {
                        byte[] byteArray2 = new byte[4];
                        r = fileInputstream.read(byteArray2, 0, 4);
                        byteBuffer = ByteArrayToNumber(byteArray2, 4, 1);
                        int temp = byteBuffer.getInt();
                        //redundant data reading
                        byte[] byteArray3 = new byte[temp];
                        r = fileInputstream.read(byteArray3, 0, temp);
                        r = fileInputstream.read(byteArray2, 0, 4);
                        subChunk2ID = new String(byteArray2);
                    }
                }
                if (i == 12) {
                    subChunk2Size = byteBuffer.getInt();
                    System.out.println(subChunk2Size);
                }
            }
//            fileInputstream.skip(44);
            bytePerSample = bitsPerSample / 8;
            float value;
            ArrayList<Float> dataVector = new ArrayList<>();
            while (true) {
                byte[] byteArray = new byte[bytePerSample];
                int v = fileInputstream.read(byteArray, 0, bytePerSample);
                value = convertToFloat(byteArray, 1);
                dataVector.add(value);
                if (v == -1) break;
            }
            float[] data = new float[dataVector.size()];
            for (int i = 0; i < dataVector.size(); i++) {
                data[i] = dataVector.get(i);
            }
            System.out.println("Total data bytes " + data.length);
            return data;
        } catch (Exception e) {
            System.out.println("Error: " + e);
            return new float[1];
        }
    }

    public float convertToFloat(byte[] array, int type) {
        ByteBuffer buffer = ByteBuffer.wrap(array);
        if (type == 1) {
            buffer.order(LITTLE_ENDIAN);
        }
        return buffer.getShort();
    }

    public ByteBuffer ByteArrayToNumber(byte[] bytes, int numOfBytes, int type) {
        ByteBuffer buffer = ByteBuffer.allocate(numOfBytes);
        if (type == 0) {
            buffer.order(BIG_ENDIAN); // Check the illustration. If it says little endian, use LITTLE_ENDIAN
        } else {
            buffer.order(LITTLE_ENDIAN);
        }
        buffer.put(bytes);
        buffer.rewind();
        return buffer;
    }

    private Float[] runFft(float[] buffer, int length) {
        Noise noise = Noise.real(length);

        float[] dst = new float[buffer.length + 2]; // real output length equals src + 2
        Float[] values = new Float[length / 2 + 1];

        dst = noise.fft(buffer, dst);

        for (int i = 0; i < dst.length / 2; i++) {
            float real = dst[i * 2];
            float imaginary = dst[i * 2 + 1];
            values[i] = (float) (Math.sqrt(Math.pow(real, 2) + Math.pow(imaginary, 2))); // / 65504.0);

//            System.out.printf("[%d] re: %.5f, im: %.5f\n", i, real, imaginary);
        }

        noise.close();

        return values;
    }
}