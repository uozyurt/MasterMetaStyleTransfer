if(__name__ == "__main__"):
    import sys
    import os
    import torch
    import cv2
    import matplotlib.pyplot as plt
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

    # test with an image
    example_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/style_layer.png"))

    # print the shape of the image before preprocess
    print(f"Example image shape before preprocess: {example_image.shape}")

    # plot the image
    plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))

    # resize the image to 256x256, it to a tensor and preprocess it (inpurt should be batch x channels x height x width)
    example_image = torch.tensor(cv2.resize(example_image, (256, 256))).permute(2,0,1).unsqueeze(0).float()/256

    # print the shape of the image after preprocess
    print(f"Example image shape: {example_image.shape}")

    # get the output of the model
    output = swin_B_first_2_stages(example_image)

    # print the shape of the outputs
    print(f"Output shape: {output.shape}")