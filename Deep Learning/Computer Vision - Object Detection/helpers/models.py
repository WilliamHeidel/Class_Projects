import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    #raise NotImplementedError('extract_peak')
    pooled = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, stride=1, padding=max_pool_ks // 2)[0,0]
    peak_mask = (heatmap == pooled) & (heatmap > min_score)
    peak_indices = peak_mask.nonzero()
    
    scores = heatmap[peak_indices[:, 0], peak_indices[:, 1]]
    top_scores, top_indices = torch.topk(scores, min(max_det, len(scores)), sorted=True)
    
    top_peak_indices = peak_indices[top_indices]
    peaks = [(float(heatmap[p[0], p[1]].item()), int(p[1]), int(p[0])) for p in top_peak_indices]

    return peaks


class Detector(torch.nn.Module):
    # Note - some ideas for implementation were found from https://github.com/milesial/Pytorch-UNet/tree/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, dropout_prob=0.25):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                        torch.nn.BatchNorm2d(n_output))
                
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
        
    class DownConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = torch.nn.Sequential(
                torch.nn.MaxPool2d(2),
                Detector.Block(in_channels, out_channels, stride=1)
            )

        def forward(self, x):
            input_size = x.size()
            fixed_size = list(input_size)
            for i in range(2, len(input_size)):
                if input_size[i] == 1:
                    fixed_size[i] = 2
            x = x.expand(fixed_size)
            return self.maxpool_conv(x)

    class UpConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = torch.nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=2, output_padding=1)
            self.conv = Detector.Block(in_channels, out_channels, stride=1)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class OutConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)
    
    def __init__(self, n_input_channels=3, n_classes=3, printout=False):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        #raise NotImplementedError('FCN.__init__')
        self.n_channels = n_input_channels
        self.n_classes = n_classes
        self.printout = printout

        self.inc = self.Block(n_input_channels, 32)
        self.down1 = self.DownConv(32, 64)
        self.down2 = self.DownConv(64, 128)
        self.down3 = self.DownConv(128, 256)
        self.up1 = self.UpConv(256, 128)
        self.up2 = self.UpConv(128, 64)
        self.up3 = self.UpConv(64, 32)
        self.outc = self.OutConv(32, n_classes)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        #raise NotImplementedError('FCN.forward')
        input = x
        if self.printout:
            print(x.size())
        x1 = self.inc(x)
        if self.printout:
            print(x1.size())
        x2 = self.down1(x1)
        if self.printout:
            print(x2.size())
        x3 = self.down2(x2)
        if self.printout:
            print(x3.size())
        x4 = self.down3(x3)
        if self.printout:
            print(x4.size())            
        x = self.up1(x4, x3)
        if self.printout:
            print(x.size())
        x = self.up2(x, x2)
        if self.printout:
            print(x.size())
        x = self.up3(x, x1)
        if self.printout:
            print(x.size())
        logits = self.outc(x)
        if self.printout:
            print(logits.size())
        return logits

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        #raise NotImplementedError('Detector.detect')
        with torch.no_grad():
            peaks_list = []
            heatmap = self.forward(image.unsqueeze(0))
            heatmap = torch.sigmoid(heatmap)
            
            for class_level in heatmap.squeeze():
                peaks = extract_peak(class_level, max_det=30)
                detections = [(float(score), int(cx), int(cy), 0.0, 0.0) for score, cx, cy in peaks]
                peaks_list.append(detections)
            
            return peaks_list


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
