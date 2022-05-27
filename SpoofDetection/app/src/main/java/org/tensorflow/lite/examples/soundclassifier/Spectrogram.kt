package org.tensorflow.lite.examples.soundclassifier

import android.util.Log
import ddf.minim.analysis.FFT

class Spectrogram(size: Int, sampleRate: Float) {
    private val TAG = this.toString()

    private var fft: FFT? = FFT(size, sampleRate)
    private var fftWidth: Int? = size
    private var specSize: Int? = null

    init {
        this.specSize = this.fft!!.specSize();
    }

    fun createSpectrogram(buffer: FloatArray) {
        val outBuffer = FloatArray(fft!!.specSize())
        fft!!.forward(buffer)

        for (wi in 0 until fft!!.specSize()) {
            outBuffer[wi] = fft!!.getBand(wi)
            Log.d(TAG, outBuffer[wi].toString())
        }

//        return TensorAudio.create(null, nul)
    }
}