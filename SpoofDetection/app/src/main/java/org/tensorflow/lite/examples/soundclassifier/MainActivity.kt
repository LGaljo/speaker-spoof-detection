/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.soundclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.WindowManager
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.core.content.ContextCompat
import androidx.core.os.HandlerCompat
import com.paramsen.noise.Noise
import org.tensorflow.lite.examples.soundclassifier.databinding.ActivityMainBinding
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.audio.classifier.AudioClassifier.AudioClassifierOptions
import kotlin.math.pow
import kotlin.math.sqrt


class MainActivity : AppCompatActivity() {
  private val probabilitiesAdapter by lazy { ProbabilitiesAdapter() }

  private var bindings: ActivityMainBinding? = null;

  private var audioClassifier: AudioClassifier? = null
  private var audioRecord: AudioRecord? = null
  private var classificationInterval = 500L // how often should classification run in milli-secs
  private lateinit var handler: Handler // background thread handler to run classification

  private var hashMap: HashMap<String, Float> = HashMap()
  var startTime: Long = System.currentTimeMillis();

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    bindings = ActivityMainBinding.inflate(layoutInflater)
    setContentView(bindings!!.root)

    bindings!!.recyclerView.apply {
        setHasFixedSize(false)
        adapter = probabilitiesAdapter
    }

    // Input switch to turn on/off classification
    keepScreenOn(bindings!!.inputSwitch.isChecked)
    bindings!!.inputSwitch.setOnCheckedChangeListener { _, isChecked ->
      if (isChecked) startAudioClassification() else stopAudioClassification()
      keepScreenOn(isChecked)
    }

    // Slider which control how often the classification task should run
    bindings!!.classificationIntervalSlider.value = classificationInterval.toFloat()
    bindings!!.classificationIntervalSlider.setLabelFormatter { value: Float ->
      "${value.toInt()} ms"
    }
    bindings!!.classificationIntervalSlider.addOnChangeListener { _, value, _ ->
      classificationInterval = value.toLong()
//        stopAudioClassification()
//        startAudioClassification()
    }
    bindings!!.recordButton.setOnClickListener {
      // Request microphone permission and start running classification
      requestMicrophonePermission()
      bindings!!.recordButton.setImageDrawable(AppCompatResources.getDrawable(applicationContext, R.drawable.ic_outline_stop_circle))
    }

    // Create a handler to run classification in a background thread
    val handlerThread = HandlerThread("backgroundThread")
    handlerThread.start()
    handler = HandlerCompat.createAsync(handlerThread.looper)

  }

  private fun startAudioClassification() {
    // If the audio classifier is initialized and running, do nothing.
    if (audioClassifier != null) return;

    // Initialize the audio classifier
    val options = AudioClassifierOptions.builder()
      .build()
    val classifier = AudioClassifier.createFromFileAndOptions(this, MODEL_FILE, options)
    val audioTensor = classifier.createInputTensorAudio()

    // Initialize the audio recorder
    val record = classifier.createAudioRecord()
    record.startRecording()

    // Define the classification runnable
    val run = object : Runnable {
      override fun run() {
        val startClassificationTime = System.currentTimeMillis()

        // Load the latest audio sample
        audioTensor.load(record)

        // FFT result
        val values: Array<Float?> = runFft(audioTensor.tensorBuffer.floatArray, 1024)
        // TODO: Determine if human speaks

        val output = classifier.classify(audioTensor)

        val filteredModelOutput = output[0].categories

        val finishClassificationTime = System.currentTimeMillis()

        Log.d(TAG, "Latency = ${finishClassificationTime - startClassificationTime}ms")
//        Log.d(TAG, "Score 0 = ${filteredModelOutput[0].score}")
//        Log.d(TAG, "Score 1 = ${filteredModelOutput[1].score}")

        if (hashMap.isEmpty()) {
          if (filteredModelOutput[0].label.equals("spoof")) {
            hashMap["spoof"] = filteredModelOutput[0].score
            hashMap["genuine"] = filteredModelOutput[1].score
          } else {
            hashMap["spoof"] = filteredModelOutput[1].score
            hashMap["genuine"] = filteredModelOutput[0].score
          }
        } else {
          if (filteredModelOutput[0].label.equals("spoof")) {
            hashMap["spoof"] = (0.9 * hashMap["spoof"]!! + 0.1 * filteredModelOutput[0].score).toFloat()
            hashMap["genuine"] = (0.9 * hashMap["genuine"]!! + 0.1 * filteredModelOutput[1].score).toFloat()
          } else {
            hashMap["spoof"] = (0.9 * hashMap["spoof"]!! + 0.1 * filteredModelOutput[1].score).toFloat()
            hashMap["genuine"] = (0.9 * hashMap["genuine"]!! + 0.1 * filteredModelOutput[0].score).toFloat()
          }
        }

        // Rerun the classification for at most 10 sec
        if (System.currentTimeMillis() - startTime < 10000) {
          handler.postDelayed(this, classificationInterval)

          // Updating the UI
          runOnUiThread {
            probabilitiesAdapter.categoryList = filteredModelOutput
            probabilitiesAdapter.notifyDataSetChanged()
          }
        } else {
          val clfs: ArrayList<Category> = ArrayList()
          clfs.add(Category("spoof", hashMap["spoof"]!!))
          clfs.add(Category("genuine", hashMap["genuine"]!!))

          // Updating the UI
          runOnUiThread {
            if (hashMap["genuine"]!! > 0.5) {
              bindings!!.layout.setBackgroundColor(getColor(android.R.color.holo_green_light))
            } else {
              bindings!!.layout.setBackgroundColor(getColor(android.R.color.holo_red_light))
            }

            probabilitiesAdapter.categoryList = clfs
            probabilitiesAdapter.notifyDataSetChanged()
          }
          bindings?.recordButton?.setImageDrawable(AppCompatResources.getDrawable(applicationContext, R.drawable.ic_outline_mic))
          stopAudioClassification()
        }
      }
    }

    // Start the classification process
    handler.post(run)

    // Save the instances we just created for use later
    audioClassifier = classifier
    audioRecord = record
  }

  private fun stopAudioClassification() {
    handler.removeCallbacksAndMessages(null)
    audioRecord?.stop()
    audioRecord = null
    audioClassifier = null
  }

  override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
    // Handles "top" resumed event on multi-window environment
    if (isTopResumedActivity && isRecordAudioPermissionGranted()) {
//      startAudioClassification()
    } else {
      stopAudioClassification()
    }
  }

  override fun onRequestPermissionsResult(
          requestCode: Int,
          permissions: Array<out String>,
          grantResults: IntArray
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == REQUEST_RECORD_AUDIO) {
      if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Log.i(TAG, "Audio permission granted :)")
        startAudioClassification()
      } else {
        Log.e(TAG, "Audio permission not granted :(")
      }
    }
  }

  @RequiresApi(Build.VERSION_CODES.M)
  private fun requestMicrophonePermission() {
    if (isRecordAudioPermissionGranted()) {
      startAudioClassification()
      startTime = System.currentTimeMillis()
      bindings!!.layout.setBackgroundColor(getColor(android.R.color.white))
    } else {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
    }
  }

  private fun isRecordAudioPermissionGranted(): Boolean {
      return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO
      ) == PackageManager.PERMISSION_GRANTED
  }

  private fun keepScreenOn(enable: Boolean) =
    if (enable) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    } else {
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

  companion object {
    const val REQUEST_RECORD_AUDIO = 1337
    private const val TAG = "AudioDemo"
    private const val MODEL_FILE = "spoof_detection_metadata.tflite"
//    private const val MODEL_FILE = "yamnet.tflite"
    private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.01f
  }

  private fun runFft(buffer: FloatArray, length: Int): Array<Float?> {
    val noise: Noise = Noise.real(length)
    var dst = FloatArray(buffer.size + 2) // real output length equals src + 2
    val values = arrayOfNulls<Float>(length / 2 + 1)

    dst = noise.fft(buffer, dst)

    for (i in 0 until dst.size / 2) {
      val real = dst[i * 2]
      val imaginary = dst[i * 2 + 1]
      values[i] = sqrt(real.toDouble().pow(2.0) + imaginary.toDouble().pow(2.0)).toFloat()

//      System.out.printf("[%d] re: %.5f, im: %.5f\n", i, real, imaginary);
    }
    noise.close()
    return values
  }
}
