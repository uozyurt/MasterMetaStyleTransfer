if(__name__ == "__main__"):
    import sys
    import os
    import torch
    import cv2
    import matplotlib.pyplot as plt
    from torchvision import transforms

    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    def apply_transform(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(image).unsqueeze(0)
    
    # add the project path to the system path
    import sys
    sys.path.append(project_absolute_path)

    # import the function to download the VGG19 model and create the cutted model
    from codes.utils import download_swin_and_create_cutted_model

    # download the model and save it
    download_swin_and_create_cutted_model(absolute_project_path = project_absolute_path,
                                          model_save_relative_path = "weights/swin_B_first_2_stages.pt")
    
    # load the model
    swin_B_first_2_stages = torch.load(os.path.join(project_absolute_path, "weights/swin_B_first_2_stages.pt"))

    # set the model to evaluation mode
    swin_B_first_2_stages.eval()


    # TODO: check if the model should be freezed or not
    # freeze the model
    for param in swin_B_first_2_stages.parameters():
        param.requires_grad = False

    # test with images
    example_image_1 = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure4/figure4_column1_content.png"))
    example_image_2 = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure4/figure4_column1_output_AdaAttN.png"))
    example_image_3 = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure4/figure4_column2_content.png"))
    



    # print the shape of the image before preprocess
    print(f"Example image shape before preprocess: {example_image_1.shape}")

    # apply the transform
    example_image_1 = apply_transform(example_image_1)

    # print the shape of the image after preprocess
    print(f"Example image shape after preprocess: {example_image_1.shape}")

    # get the output of the model
    output = swin_B_first_2_stages(example_image_1)

    # print the shape of the outputs
    print(f"Output shape of swin_S: {output.shape}")

    # preprocess the second and third images
    example_image_2 = apply_transform(example_image_2)
    example_image_3 = apply_transform(example_image_3)

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