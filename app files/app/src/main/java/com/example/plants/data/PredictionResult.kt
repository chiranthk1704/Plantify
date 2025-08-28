package com.example.plants.data

// Sealed class to represent different states of prediction
sealed class PredictionResult {
    object Loading : PredictionResult()
    data class Success(
        val resnetPrediction: ModelPrediction,
        val yoloPrediction: ModelPrediction
    ) : PredictionResult()
    data class Error(val message: String) : PredictionResult()
}

// Data class for individual model prediction
data class ModelPrediction(
    val className: String,
    val confidence: String,
    val timeTaken: String,
    val memoryUsage: String
)

// Data classes for API communication
data class PredictionRequest(val image: String)

data class PredictionResponse(
    val resnet_prediction: ModelPrediction,
    val yolo_prediction: ModelPrediction
)