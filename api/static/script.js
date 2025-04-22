document.addEventListener("DOMContentLoaded", () => {
	const uploadForm = document.getElementById("uploadForm");
	const statusSection = document.getElementById("statusSection");
	const resultSection = document.getElementById("resultSection");
	const progressBar = document.getElementById("progressBar");
	const statusText = document.getElementById("statusText");
	const resultVideo = document.getElementById("resultVideo");
	const resultImage = document.getElementById("resultImage");
	const downloadLink = document.getElementById("downloadLink");

	let currentTaskId = null;
	let statusCheckInterval = null;

	uploadForm.addEventListener("submit", async (e) => {
		e.preventDefault();

		const videoFile = document.getElementById("videoFile").files[0];
		const audioFile = document.getElementById("audioFile").files[0];

		if (!videoFile || !audioFile) {
			alert("Please select both video and audio files");
			return;
		}

		const formData = new FormData();
		formData.append("video", videoFile);
		formData.append("audio", audioFile);

		try {
			// Show status section
			statusSection.style.display = "block";
			resultSection.style.display = "none";
			progressBar.style.width = "0%";
			statusText.textContent = "Uploading files...";

			// Upload files
			const response = await fetch("/lipsync/process", {
				method: "POST",
				body: formData,
			});

			if (!response.ok) {
				throw new Error("Failed to start processing");
			}

			const data = await response.json();
			currentTaskId = data.task_id;

			// Start checking status
			startStatusCheck();
		} catch (error) {
			console.error("Error:", error);
			statusText.textContent = `Error: ${error.message}`;
			progressBar.style.backgroundColor = "var(--error-color)";
		}
	});

	function startStatusCheck() {
		if (statusCheckInterval) {
			clearInterval(statusCheckInterval);
		}

		statusCheckInterval = setInterval(async () => {
			try {
				const response = await fetch(
					`/lipsync/status/${currentTaskId}`
				);
				if (!response.ok) {
					throw new Error("Failed to get status");
				}

				const data = await response.json();

				// Update progress
				progressBar.style.width = `${data.progress}%`;
				statusText.textContent = `Status: ${data.status}`;

				// Handle completion
				if (data.status === "completed") {
					clearInterval(statusCheckInterval);
					showResult(data.output_path);
				} else if (data.status === "failed") {
					clearInterval(statusCheckInterval);
					statusText.textContent = `Error: ${
						data.error || "Processing failed"
					}`;
					progressBar.style.backgroundColor = "var(--error-color)";
				}
			} catch (error) {
				console.error("Error checking status:", error);
				clearInterval(statusCheckInterval);
				statusText.textContent = `Error: ${error.message}`;
				progressBar.style.backgroundColor = "var(--error-color)";
			}
		}, 2000); // Check every 2 seconds
	}

	function showResult(outputPath) {
		resultSection.style.display = "block";
		statusText.textContent = "Processing complete!";

		// Determine if the output is a video or image
		const isVideo =
			outputPath.toLowerCase().endsWith(".mp4") ||
			outputPath.toLowerCase().endsWith(".webm") ||
			outputPath.toLowerCase().endsWith(".mov");

		if (isVideo) {
			resultVideo.style.display = "block";
			resultImage.style.display = "none";
			resultVideo.src = outputPath;
		} else {
			resultVideo.style.display = "none";
			resultImage.style.display = "block";
			resultImage.src = outputPath;
		}

		// Set up download link
		downloadLink.href = outputPath;
		downloadLink.download = outputPath.split("/").pop();
	}
});
