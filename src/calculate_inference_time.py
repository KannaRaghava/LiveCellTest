import time


def calculate_inference_time():

    # Set the model to evaluation mode
    model.eval()

    # Create a list to store inference times
    inference_times = []

    # Iterate over the evaluation dataset and measure inference time
    for imgs, _ in data_loader:
        imgs = [img.to(device) for img in imgs]

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            _ = model(imgs)
        end_time = time.time()

        # Calculate and record the inference time for this batch
        batch_inference_time = end_time - start_time
        inference_times.append(batch_inference_time)

    # Calculate and print the average inference time
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f'Average Inference Time: {average_inference_time:.4f} seconds')

    # Now, you can compare the training and inference times
    print(f'Inference Time: {average_inference_time:.4f} seconds')
