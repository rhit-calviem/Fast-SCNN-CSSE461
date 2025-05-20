import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from utils.visualize import get_color_pallete

def parse_args():
    parser = argparse.ArgumentParser(description='Fast-SCNN Demo')
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='Model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='Dataset name (default: citys)')
    parser.add_argument('--input-pic', type=str, required=True,
                        help='Path to the input image')
    # Added parser for model, otherwise it only ever used the original weights
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to the .pth model weights')
    parser.add_argument('--outdir', type=str, default='test_result',
                        help='Directory to save the result image')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only')
    return parser.parse_args()

def demo():
    args = parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    # Load model, set boolean to False to use other models besides original
    model = get_fast_scnn(args.dataset, pretrained=False).to(device)
    print(f"[INFO] Loading model from {args.resume}")
    model.load_state_dict(torch.load(args.resume, map_location=device))
    model.eval()

    # Load image
    input_image = Image.open(args.input_pic).convert('RGB')
    # Added preprocessing step to make sure other images could also work
    input_image = input_image.resize((2048, 1024), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    image_tensor = transform(input_image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output[0], 1).squeeze(0).cpu().numpy()

    # Convert prediction to color mask
    mask = get_color_pallete(pred, args.dataset)

    # Save result
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    image_name = os.path.basename(args.input_pic).replace('.jpg', '').replace('.png', '')
    save_path = os.path.join(args.outdir, f'{image_name}_segmentation.png')
    mask.save(save_path)
    print(f"[INFO] Saved segmentation result to {save_path}")

if __name__ == '__main__':
    demo()
