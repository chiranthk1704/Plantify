package com.example.plants.data

import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

object RetrofitClient {
    // Replace with your actual PythonAnywhere URL
    private const val BASE_URL = "https://chirag97563.pythonanywhere.com/"

    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val plantPredictionApi: PlantPredictionApi = retrofit.create(PlantPredictionApi::class.java)
}

// API Interface
interface PlantPredictionApi {
    @POST("predict")
    suspend fun getPrediction(@Body requestBody: PredictionRequest): Response<PredictionResponse>
}
