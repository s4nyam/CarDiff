import os
from argparse import ArgumentParser
# python main.py --segmentation_guided
# python main.py --cardiff --segmentation_guided   (for CarDiff)
# torch imports
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# HF imports
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets

# custom imports
from training import TrainingConfig, train_loop
from eval import evaluate_generation, evaluate_sample_many

# CarDiff imports
from cardiff import (
    CarDiffModel,
    CarDiffTrainingConfig,
    cardiff_train_loop,
    CarDiffPipeline,
)

def main(
    mode,
    img_size,
    num_img_channels,
    dataset,
    img_dir,
    seg_dir,
    model_type,
    segmentation_guided,
    segmentation_channel_mode,
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    resume_epoch=None,
    use_ablated_segmentations=False,
    eval_shuffle_dataloader=True,

    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000
):
    #GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device))

    # load config
    output_dir = '{}-{}-{}'.format(model_type.lower(), dataset, img_size)  # the model namy locally and on the HF Hub
    if segmentation_guided:
        output_dir += "-segguided"
        assert seg_dir is not None, "must provide segmentation directory for segmentation guided training/sampling"

    if use_ablated_segmentations or eval_mask_removal or eval_blank_mask:
        output_dir += "-ablated"

    print("output dir: {}".format(output_dir))

    if mode == "train":
        evalset_name = "val"
        assert img_dir is not None, "must provide image directory for training"
    elif "eval" in mode:
        evalset_name = "test"

    print("using evaluation set: {}".format(evalset_name))

    config = TrainingConfig(
        image_size = img_size,
        dataset = dataset,
        segmentation_guided = segmentation_guided,
        segmentation_channel_mode = segmentation_channel_mode,
        num_segmentation_classes = num_segmentation_classes,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        num_epochs = num_epochs,
        output_dir = output_dir,
        model_type=model_type,
        resume_epoch=resume_epoch,
        use_ablated_segmentations=use_ablated_segmentations
    )

    load_images_as_np_arrays = False
    if num_img_channels not in [1, 3]:
        load_images_as_np_arrays = True
        print("image channels not 1 or 3, attempting to load images as np arrays...")

    if config.segmentation_guided:
        seg_types = os.listdir(seg_dir)
        seg_paths_train = {} 
        seg_paths_eval = {}

        # train set
        if img_dir is not None: 
            # make sure the images are matched to the segmentation masks
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train)]
            for seg_type in seg_types:
                seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(img_dir_train)]
        else:
            for seg_type in seg_types:
                seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(os.path.join(seg_dir, seg_type, "train"))]

        # eval set
        if img_dir is not None: 
            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = [os.path.join(img_dir_eval, f) for f in os.listdir(img_dir_eval)]
            for seg_type in seg_types:
                seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, evalset_name, f) for f in os.listdir(img_dir_eval)]
        else:
            for seg_type in seg_types:
                seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, evalset_name, f) for f in os.listdir(os.path.join(seg_dir, seg_type, evalset_name))]

        if img_dir is not None:
            dset_dict_train = {
                    **{"image": img_paths_train},
                    **{"seg_{}".format(seg_type): seg_paths_train[seg_type] for seg_type in seg_types}
                }
            
            dset_dict_eval = {
                    **{"image": img_paths_eval},
                    **{"seg_{}".format(seg_type): seg_paths_eval[seg_type] for seg_type in seg_types}
            }
        else:
            dset_dict_train = {
                    **{"seg_{}".format(seg_type): seg_paths_train[seg_type] for seg_type in seg_types}
                }
            
            dset_dict_eval = {
                    **{"seg_{}".format(seg_type): seg_paths_eval[seg_type] for seg_type in seg_types}
            }


        if img_dir is not None:
            # add image filenames to dataset
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["image"]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["image"]]
        else:
            # use segmentation filenames as image filenames
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["seg_{}".format(seg_types[0])]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["seg_{}".format(seg_types[0])]]

        dataset_train = datasets.Dataset.from_dict(dset_dict_train)
        dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

        # load the images
        if not load_images_as_np_arrays and img_dir is not None:
            dataset_train = dataset_train.cast_column("image", datasets.Image())
            dataset_eval = dataset_eval.cast_column("image", datasets.Image())

        for seg_type in seg_types:
            dataset_train = dataset_train.cast_column("seg_{}".format(seg_type), datasets.Image())

        for seg_type in seg_types:
            dataset_eval = dataset_eval.cast_column("seg_{}".format(seg_type), datasets.Image())

    else:
        if img_dir is not None:
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train)]

            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = [os.path.join(img_dir_eval, f) for f in os.listdir(img_dir_eval)]

            dset_dict_train = {
                    **{"image": img_paths_train}
                }

            dset_dict_eval = {
                    **{"image": img_paths_eval}
                }

            # add image filenames to dataset
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["image"]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["image"]]

            dataset_train = datasets.Dataset.from_dict(dset_dict_train)
            dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

            # load the images
            if not load_images_as_np_arrays:
                dataset_train = dataset_train.cast_column("image", datasets.Image())
                dataset_eval = dataset_eval.cast_column("image", datasets.Image())

    # training set preprocessing
    if not load_images_as_np_arrays:
        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                # transforms.RandomHorizontalFlip(), # flipping wouldn't result in realistic images
                transforms.ToTensor(),
                transforms.Normalize(
                    num_img_channels * [0.5], 
                    num_img_channels * [0.5]),
            ]
        )
    else:
        # resizing will be done in the transform function
        preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    num_img_channels * [0.5], 
                    num_img_channels * [0.5]),
            ]
        )

    if num_img_channels == 1:
        PIL_image_type = "L"
    elif num_img_channels == 3:
        PIL_image_type = "RGB"
    else:
        PIL_image_type = None

    if config.segmentation_guided:
        preprocess_segmentation = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
        )

        def transform(examples):
            if img_dir is not None:
                if not load_images_as_np_arrays:
                    images = [preprocess(image.convert(PIL_image_type)) for image in examples["image"]]
                else:
                    # load np array as torch tensor, resize, then normalize
                    images = [
                        preprocess(F.interpolate(torch.tensor(np.load(image)).unsqueeze(0).float(), size=(config.image_size, config.image_size)).squeeze()) for image in examples["image"]
                        ]

            images_filenames = examples["image_filename"]

            segs = {}
            for seg_type in seg_types:
                segs["seg_{}".format(seg_type)] = [preprocess_segmentation(image.convert("L")) for image in examples["seg_{}".format(seg_type)]]
            # return {"images": images, "seg_breast": seg_breast, "seg_dv": seg_dv}
            if img_dir is not None:
                return {**{"images": images}, **segs, **{"image_filenames": images_filenames}}
            else:
                return {**segs, **{"image_filenames": images_filenames}}
            
        dataset_train.set_transform(transform)
        dataset_eval.set_transform(transform)

    else:
        if img_dir is not None:
            def transform(examples):
                if not load_images_as_np_arrays:
                    images = [preprocess(image.convert(PIL_image_type)) for image in examples["image"]]
                else:
                    images = [
                        preprocess(F.interpolate(torch.tensor(np.load(image)).unsqueeze(0).float(), size=(config.image_size, config.image_size)).squeeze()) for image in examples["image"]
                        ]
                images_filenames = examples["image_filename"]
                #return {"images": images, "image_filenames": images_filenames}
                return {"images": images, **{"image_filenames": images_filenames}}
        
            dataset_train.set_transform(transform)
            dataset_eval.set_transform(transform)

    if ((img_dir is None) and (not segmentation_guided)):
        train_dataloader = None
        # just make placeholder dataloaders to iterate through when sampling from uncond model
        eval_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(config.eval_batch_size, num_img_channels, config.image_size, config.image_size)),
            batch_size=config.eval_batch_size,
            shuffle=eval_shuffle_dataloader
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
                dataset_train, 
                batch_size=config.train_batch_size, 
                shuffle=True
                )

        eval_dataloader = torch.utils.data.DataLoader(
                dataset_eval, 
                batch_size=config.eval_batch_size, 
                shuffle=eval_shuffle_dataloader
                )

    # define the model
    in_channels = num_img_channels
    if config.segmentation_guided:
        assert config.num_segmentation_classes is not None
        assert config.num_segmentation_classes > 1, "must have at least 2 segmentation classes (INCLUDING background)" 
        if config.segmentation_channel_mode == "single":
            in_channels += 1
        elif config.segmentation_channel_mode == "multi":
            in_channels = len(seg_types) + in_channels

    model = diffusers.UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=in_channels,  # the number of input channels, 3 for RGB images
        out_channels=num_img_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )

    if (mode == "train" and resume_epoch is not None) or "eval" in mode:
        if mode == "train":
            print("resuming from model at training epoch {}".format(resume_epoch))
        elif "eval" in mode:
            print("loading saved model...")
        model = model.from_pretrained(os.path.join(config.output_dir, 'unet'), use_safetensors=True)

    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    if mode == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        # train
        train_loop(
            config, 
            model, 
            noise_scheduler, 
            optimizer, 
            train_dataloader, 
            eval_dataloader, 
            lr_scheduler, 
            device=device
            )
    elif mode == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        evaluate_generation(
            config, 
            model, 
            noise_scheduler,
            eval_dataloader, 
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device
            )

    elif mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device
            )

    else:
        raise ValueError("mode \"{}\" not supported.".format(mode))


# ======================================================================
# CarDiff entry point
# ======================================================================

def main_cardiff(
    mode,
    img_size,
    num_img_channels,
    dataset,
    img_dir,
    seg_dir,
    model_type,
    segmentation_channel_mode,
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    resume_epoch=None,
    eval_shuffle_dataloader=True,
    eval_sample_size=1000,
    vae_pretrained_path=None,
    enable_path_b=True,
    path_b_start_epoch=10,
):
    """
    CarDiff training / evaluation entry point.
    
    Builds a CarDiffModel, sets up dataloaders via the same pipeline as the
    baseline, and dispatches to cardiff_train_loop or CarDiffPipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CarDiff] running on {device}")

    output_dir = f"cardiff-{dataset}-{img_size}"
    print(f"[CarDiff] output dir: {output_dir}")

    evalset_name = "val" if mode == "train" else "test"
    print(f"[CarDiff] evaluation set: {evalset_name}")

    # ------------------------------------------------------------------
    # Build dataloaders (reuse existing logic)
    # ------------------------------------------------------------------
    assert seg_dir is not None, "CarDiff requires --seg_dir"

    seg_types = os.listdir(seg_dir)

    def _build_paths(img_base, seg_base, split):
        img_split = os.path.join(img_base, split) if img_base else None
        img_paths = sorted([os.path.join(img_split, f) for f in os.listdir(img_split)]) if img_split else None
        seg_paths = {}
        for st in seg_types:
            ref = os.listdir(img_split) if img_split else os.listdir(os.path.join(seg_base, st, split))
            seg_paths[st] = sorted([os.path.join(seg_base, st, split, f) for f in ref])
        return img_paths, seg_paths

    img_paths_train, seg_paths_train = _build_paths(img_dir, seg_dir, "train")
    img_paths_eval, seg_paths_eval = _build_paths(img_dir, seg_dir, evalset_name)

    def _to_dataset(img_paths, seg_paths):
        d = {}
        if img_paths:
            d["image"] = img_paths
            d["image_filename"] = [os.path.basename(f) for f in img_paths]
        else:
            first_key = list(seg_paths.keys())[0]
            d["image_filename"] = [os.path.basename(f) for f in seg_paths[first_key]]
        for st in seg_types:
            d[f"seg_{st}"] = seg_paths[st]
        ds = datasets.Dataset.from_dict(d)
        if img_paths:
            ds = ds.cast_column("image", datasets.Image())
        for st in seg_types:
            ds = ds.cast_column(f"seg_{st}", datasets.Image())
        return ds

    dataset_train = _to_dataset(img_paths_train, seg_paths_train)
    dataset_eval = _to_dataset(img_paths_eval, seg_paths_eval)

    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(num_img_channels * [0.5], num_img_channels * [0.5]),
    ])
    preprocess_seg = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    pil_mode = "L" if num_img_channels == 1 else "RGB"

    def transform(examples):
        out = {}
        if "image" in examples:
            out["images"] = [preprocess(img.convert(pil_mode)) for img in examples["image"]]
        out["image_filenames"] = examples["image_filename"]
        for st in seg_types:
            out[f"seg_{st}"] = [preprocess_seg(img.convert("L")) for img in examples[f"seg_{st}"]]
        return out

    dataset_train.set_transform(transform)
    dataset_eval.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=train_batch_size, shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset_eval, batch_size=eval_batch_size, shuffle=eval_shuffle_dataloader
    )

    # ------------------------------------------------------------------
    # Build CarDiff model
    # ------------------------------------------------------------------
    cardiff_config = CarDiffTrainingConfig(
        image_size=img_size,
        img_channels=num_img_channels,
        num_classes=num_segmentation_classes,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        model_type=model_type,
        output_dir=output_dir,
        resume_epoch=resume_epoch,
        vae_pretrained_path=vae_pretrained_path,
        enable_path_b=enable_path_b,
        path_b_start_epoch=path_b_start_epoch,
        dataset=dataset,
    )

    model = CarDiffModel(
        image_size=img_size,
        img_channels=num_img_channels,
        num_classes=num_segmentation_classes,
        vae_pretrained_path=vae_pretrained_path,
    )

    # Noise scheduler
    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Resume from checkpoint
    if resume_epoch is not None:
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{resume_epoch}", "cardiff_model.pt")
        if os.path.exists(ckpt_path):
            print(f"[CarDiff] Resuming from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)

    model = model.to(device)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if mode == "train":
        cardiff_train_loop(
            cardiff_config,
            model,
            noise_scheduler,
            train_dataloader,
            eval_dataloader,
            device=str(device),
        )

    elif mode == "eval" or mode == "eval_many":
        pipeline = CarDiffPipeline(model, noise_scheduler, device=str(device))

        sample_dir = os.path.join(output_dir, f"samples_cardiff_{eval_sample_size}")
        os.makedirs(sample_dir, exist_ok=True)

        num_sampled = 0
        for batch in eval_dataloader:
            if num_sampled >= eval_sample_size:
                break

            # Build single-channel mask
            seg_keys = [k for k in batch.keys() if k.startswith("seg_")]
            masks = torch.zeros_like(batch[seg_keys[0]]).to(device)
            for k in seg_keys:
                seg = batch[k].to(device)
                masks[masks == 0] = seg[masks == 0]

            images = pipeline(masks, num_inference_steps=1000, output_type="pil")
            for i, img in enumerate(images):
                fname = batch["image_filenames"][i]
                img.save(os.path.join(sample_dir, f"cardiff_{fname}"))
                num_sampled += 1
                if num_sampled >= eval_sample_size:
                    break
            print(f"Sampled {num_sampled}/{eval_sample_size}")

        print(f"[CarDiff] Sampling complete. Saved to {sample_dir}")

    else:
        raise ValueError(f"mode \"{mode}\" not supported for CarDiff.")


if __name__ == "__main__":
    # parse args:
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train') # eval_many or train
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_img_channels', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="dc")
    parser.add_argument('--img_dir', type=str, default="DATA_FOLDER")
    parser.add_argument('--seg_dir', type=str, default="MASK_FOLDER")
    parser.add_argument('--model_type', type=str, default="DDPM")
    parser.add_argument('--segmentation_guided', action='store_true', help='use segmentation guided training/sampling?')
    parser.add_argument('--segmentation_channel_mode', type=str, default="single", help='single == all segmentations in one channel, multi == each segmentation in its own channel')
    parser.add_argument('--num_segmentation_classes', type=int, default=4, help='number of segmentation classes, including background')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--resume_epoch', type=int, default=780, help='resume training starting at this epoch')

    # novel options
    parser.add_argument('--use_ablated_segmentations', action='store_true', help='use mask ablated training and any evaluation? sometimes randomly remove class(es) from mask during training and sampling.')

    # other options
    parser.add_argument('--eval_noshuffle_dataloader', action='store_true', help='if true, don\'t shuffle the eval dataloader')

    # args only used in eval
    parser.add_argument('--eval_mask_removal', action='store_true', help='if true, evaluate gradually removing anatomies from mask and re-sampling')
    parser.add_argument('--eval_blank_mask', action='store_true', help='if true, evaluate sampling conditioned on blank (zeros) masks')
    parser.add_argument('--eval_sample_size', type=int, default=100, help='number of images to sample when using eval_many mode')

    # CarDiff options
    parser.add_argument('--cardiff', action='store_true', help='use CarDiff (causally-structured diffusion) instead of baseline')
    parser.add_argument('--vae_pretrained_path', type=str, default=None, help='path to pre-trained VAE weights for CarDiff')
    parser.add_argument('--enable_path_b', action='store_true', help='enable self-supervised Path B in CarDiff')
    parser.add_argument('--path_b_start_epoch', type=int, default=10, help='epoch to start Path B training')

    args = parser.parse_args()

    if args.cardiff:
        # ---- CarDiff pathway ----
        main_cardiff(
            args.mode,
            args.img_size,
            args.num_img_channels,
            args.dataset,
            args.img_dir,
            args.seg_dir,
            args.model_type,
            args.segmentation_channel_mode,
            args.num_segmentation_classes,
            args.train_batch_size,
            args.eval_batch_size,
            args.num_epochs,
            args.resume_epoch,
            not args.eval_noshuffle_dataloader,
            args.eval_sample_size,
            args.vae_pretrained_path,
            args.enable_path_b,
            args.path_b_start_epoch,
        )
    else:
        # ---- Baseline pathway (unchanged) ----
        main(
            args.mode,
            args.img_size,
            args.num_img_channels,
            args.dataset,
            args.img_dir,
            args.seg_dir,
            args.model_type,
            args.segmentation_guided,
            args.segmentation_channel_mode,
            args.num_segmentation_classes,
            args.train_batch_size,
            args.eval_batch_size,
            args.num_epochs,
            args.resume_epoch,
            args.use_ablated_segmentations,
            not args.eval_noshuffle_dataloader,
            # args only used in eval
            args.eval_mask_removal,
            args.eval_blank_mask,
            args.eval_sample_size,
        )
