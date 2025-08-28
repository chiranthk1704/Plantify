package com.example.plants.data

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Base64
import java.io.ByteArrayOutputStream
import java.io.IOException

class PlantRepository(private val context: Context) {
    private val api = RetrofitClient.plantPredictionApi

    suspend fun getPrediction(imageUri: Uri): PredictionResponse {
        // Convert image to base64
        val base64Image = getBase64FromUri(imageUri)

        // Make API call
        val response = api.getPrediction(PredictionRequest(base64Image))
        if (!response.isSuccessful) {
            throw Exception("Prediction failed: ${response.errorBody()?.string()}")
        }
        return response.body() ?: throw Exception("Empty response")
    }

    private fun getBase64FromUri(uri: Uri): String {
        val inputStream = context.contentResolver.openInputStream(uri)
        val bytes = inputStream?.readBytes() ?: throw IOException("Failed to read image")
        return Base64.encodeToString(bytes, Base64.DEFAULT)
    }

    // Optional: Add image compression method
    private fun compressImage(bitmap: Bitmap, quality: Int = 70): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val compressedBytes = outputStream.toByteArray()
        return Base64.encodeToString(compressedBytes, Base64.DEFAULT)
    }
}