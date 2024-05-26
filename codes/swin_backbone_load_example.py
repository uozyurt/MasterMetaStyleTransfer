if(__name__ == "__main__"):
    import sys
    import os
    import torch
    import cv2
    import matplotlib.pyplot as plt
    from torchvision import transforms

    import sys
    # add the project path to the system path
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_absolute_path)

    # import the function to download the swin model and create the cutted model
    from codes.utils import download_swin_and_create_cutted_model


    # define the transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with mean and std
    ])
    def apply_transform(image):
        return transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    
    
    # test with images
    example_image_1 = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure4/figure4_column1_content.png"))
    example_image_2 = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure4/figure4_column1_output_AdaAttN.png"))
    example_image_3 = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure4/figure4_column2_content.png"))


    # print the shape of the image before preprocess
    print(f"Example image shape before preprocess: {example_image_1.shape}")

    # preprocess the second images
    example_image_1 = apply_transform(example_image_1)
    example_image_2 = apply_transform(example_image_2)
    example_image_3 = apply_transform(example_image_3)
    # print the shape of the image after preprocess
    print(f"Example image shape after preprocess: {example_image_1.shape}")



    for swin_variant in ["swin_T", "swin_S", "swin_B"]:

        # get the current relative path for the swin model
        swin_model_relative_path = os.path.join("weights", f"{swin_variant}_first_2_stages.pt")

        # download the model and save it
        download_swin_and_create_cutted_model(absolute_project_path = project_absolute_path,
                                            model_save_relative_path = swin_model_relative_path)
        
        # load the model
        swin_B_first_2_stages = torch.load(os.path.join(project_absolute_path, swin_model_relative_path))

        # set the model to evaluation mode
        swin_B_first_2_stages.eval()




        with torch.no_grad():
            # get the output of the model
            output = swin_B_first_2_stages(example_image_1)

        # print the shape of the outputs
        print(f"Output shape of {swin_variant}: {output.shape}")



        # get the output of the model
        output_2 = swin_B_first_2_stages(example_image_2)
        output_3 = swin_B_first_2_stages(example_image_3)

        # permute
        output = output.permute(0, 2, 3, 1)
        output_2 = output_2.permute(0, 2, 3, 1)
        output_3 = output_3.permute(0, 2, 3, 1)

        # get the cosine similarity between the outputs
        similarity_1_2 = torch.nn.functional.cosine_similarity(output, output_2)
        similarity_1_3 = torch.nn.functional.cosine_similarity(output, output_3)

        # print the cosine similarity
        print(f"Similarity between the first and second image: {torch.mean(similarity_1_2)}")
        print(f"Similarity between the first and third image: {torch.mean(similarity_1_3)}")

        print("\n\n")