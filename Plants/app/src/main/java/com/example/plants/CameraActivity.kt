package com.example.plants

import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton

class CameraActivity : AppCompatActivity() {

    private lateinit var capturedImageView: ImageView
    private var imageUri: Uri? = null
    private lateinit var takePictureLauncher: androidx.activity.result.ActivityResultLauncher<Uri>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        capturedImageView = findViewById(R.id.imageView)
        val cameraButton: MaterialButton = findViewById(R.id.button_camera)
        val analyzeButton: MaterialButton = findViewById(R.id.button_analyze)
        val homebutton: MaterialButton = findViewById(R.id.button_back)
        // Setup for taking a picture with proper permissions
        takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                capturedImageView.setImageURI(imageUri)
                Toast.makeText(this, "Image captured", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Image capture failed", Toast.LENGTH_SHORT).show()
            }
        }

        // Handle camera button click
        cameraButton.setOnClickListener {
            if (checkCameraPermission()) {
                openCamera()
            }
        }

        analyzeButton.setOnClickListener {
            if (imageUri != null) {
                Toast.makeText(this, "Proceed to analyze", Toast.LENGTH_SHORT).show()
                val intent = Intent(this, ResultActivity::class.java)
                intent.putExtra("IMAGE_URI", imageUri.toString())
                startActivity(intent)
            } else {
                Toast.makeText(this, "Please capture an image first", Toast.LENGTH_SHORT).show()
            }
        }
        homebutton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
        }

    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun openCamera() {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.TITLE, "CameraCapture")
            put(MediaStore.Images.Media.DESCRIPTION, "Captured by Camera")
        }
        imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)

        imageUri?.let {
            takePictureLauncher.launch(it)
        } ?: Toast.makeText(this, "Could not prepare for camera", Toast.LENGTH_SHORT).show()
    }
}
