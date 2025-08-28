package com.example.plants

import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.example.plants.databinding.ActivityResultBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

class ResultActivity : AppCompatActivity() {
    private lateinit var binding: ActivityResultBinding
    private lateinit var viewModel: ResultViewModel
    private var imageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel = ViewModelProvider(this)[ResultViewModel::class.java]

        val imageUriString = intent.getStringExtra("IMAGE_URI")
        if (imageUriString != null) {
            imageUri = Uri.parse(imageUriString)
            try {
                binding.imageViewResult.setImageURI(imageUri)
                binding.textPredictionResult.text = "Click 'Start Analysis' to begin plant detection"

                binding.buttonStart.apply {
                    isEnabled = true
                    setOnClickListener {
                        imageUri?.let { uri ->
                            startAnalysis(uri)
                            isEnabled = false
                        }
                    }
                }
            } catch (e: Exception) {
                Toast.makeText(this, "Unable to load image", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "No image to display", Toast.LENGTH_SHORT).show()
        }

        binding.buttonBack.setOnClickListener {
            finish()
        }
    }

    private fun startAnalysis(imageUri: Uri) {
        lifecycleScope.launch {
            try {
                binding.progressBar.visibility = View.VISIBLE
                binding.textPredictionResult.text = "Sending image for analysis..."

                viewModel.analyzeImage(imageUri, applicationContext).collect { result ->
                    when (result) {
                        is AnalysisResult.Success -> {
                            binding.textPredictionResult.text = """
                                ResNet-9 Results:
                                Plant: ${result.resnetResults[0]}
                                Confidence: ${result.resnetResults[1]}%
                                Time Taken: ${result.resnetResults[2]} sec
                                Memory Usage: ${result.resnetResults[3]} MB
                                
                                YOLO Results:
                                Plant: ${result.yoloResults[0]}
                                Confidence: ${result.yoloResults[1]}%
                                Time Taken: ${result.yoloResults[2]} sec
                                Memory Usage: ${result.yoloResults[3]} MB
                            """.trimIndent()
                            binding.buttonStart.isEnabled = true
                        }
                        is AnalysisResult.Error -> {
                            binding.textPredictionResult.text = "Error: ${result.message}"
                            binding.buttonStart.isEnabled = true
                        }
                        is AnalysisResult.Loading -> {
                            binding.textPredictionResult.text = "Processing image..."
                        }
                    }
                    binding.progressBar.visibility = View.GONE
                }
            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                binding.textPredictionResult.text = "Error: ${e.message}"
                binding.buttonStart.isEnabled = true
            }
        }
    }
}

sealed class AnalysisResult {
    object Loading : AnalysisResult()
    data class Success(
        val resnetResults: List<String>,
        val yoloResults: List<String>
    ) : AnalysisResult()
    data class Error(val message: String) : AnalysisResult()
}

class ResultViewModel : ViewModel() {
    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    fun analyzeImage(imageUri: Uri, context: android.content.Context) = flow {
        emit(AnalysisResult.Loading)
        try {
            // Create temporary file from URI
            val inputStream = context.contentResolver.openInputStream(imageUri)
            val tempFile = File.createTempFile("upload", ".jpg", context.cacheDir)

            FileOutputStream(tempFile).use { outputStream ->
                inputStream?.copyTo(outputStream)
            }

            // Send POST request with image
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    "image.jpg",
                    tempFile.asRequestBody("image/*".toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url("http://192.168.216.46:5000/predict")  // Replace with your IP address
                .post(requestBody)
                .build()

            // Execute POST request and handle response
            val response = client.newCall(request).execute()

            if (response.isSuccessful) {
                val jsonResponse = JSONObject(response.body?.string() ?: "{}")
                val resnetResults = jsonResponse.getJSONArray("ResNet-9").let { array ->
                    List(array.length()) { array.getString(it) }
                }
                val yoloResults = jsonResponse.getJSONArray("YOLO").let { array ->
                    List(array.length()) { array.getString(it) }
                }
                emit(AnalysisResult.Success(resnetResults, yoloResults))
            } else {
                emit(AnalysisResult.Error("Server error: ${response.code}"))
            }

            // Clean up
            tempFile.delete()

        } catch (e: Exception) {
            emit(AnalysisResult.Error(e.message ?: "Unknown error occurred"))
        }
    }.flowOn(Dispatchers.IO)
}