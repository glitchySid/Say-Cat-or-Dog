$(document).ready(function() {
    var dropzone = $("#dropzone");

    // ... (rest of the drop handling code remains the same)

    function sendImageForPrediction(imageData) {
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: { image: imageData },
            success: function(response) {
                $("#prediction").text("Predicted Class: " + response.class);
                $("#confidence").text("Confidence: " + response.confidence.toFixed(2));  // Format confidence to 2 decimal places
            },
            error: function(error) {
                console.error(error);
                $("#prediction").text("Error: Could not make prediction");
            }
        });
    }
});
