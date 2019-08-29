<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>AI Model Zoo</h1>
    </td>
 </tr>
 </table>

# Introduction
This repository includes optimized deep learning models to speed up the deployment of deep learning inference on Xilinx&trade; platforms. These models cover different applications, including but not limited to ADAS/AD, video surveillance, robotics, data center, etc. You can get started with these free pre-trained models to enjoy the benefits of deep learning acceleration.

![Missing Image:xlnx_model_zoo.png](images/xlnx_model_zoo.png)

## Model Information
The following table includes comprehensive information about each model, including application, framework, training and validation dataset, backbone, input size, computation as well as float and fixed-point precision. 

<details>
 <summary><b>Click here to view details</b></summary>
 
| No\. | Application              | Model                          | Name                                                | Framework  | Backbone       | Input Size | OPS per image | Training Set                            | Val Set                 | Float \(Top1, Top5\)/ mAP/mIoU | Fixed \(Top1, Top5\)/mAP/mIoU |
|------|--------------------------|--------------------------------|-----------------------------------------------------|------------|----------------|------------|---------------|-----------------------------------------|-------------------------|--------------------------------|-------------------------------|
| 1    | Image Classification     | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | caffe      | resnet50       | 224\*224   | 7\.7G         | ImageNet Train                          | ImageNet Validataion    | 0\.74828/0\.92135              | 0\.7338/0\.9130               |
| 2    | Image Classification     | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | caffe      | inception\_v1  | 224\*224   | 3\.16G        | ImageNet Train                          | ImageNet Validataion    | 0\.689/0\.897                  | 0\.69882/0\.894122            |
| 3    | Image Classification     | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | caffe      | bn\-inception  | 224\*224   | 4G            | ImageNet Train                          | ImageNet Validataion    | 0\.7283/0\.9109                | 0\.7170/0\.9033               |
| 4    | Image Classification     | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | caffe      | inception\_v3  | 299\*299   | 11\.4G        | ImageNet Train                          | ImageNet Validataion    | 0\.77058/0\.93326              | 0\.76264/0\.930322            |
| 5    | Image Classification     | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | caffe      | MobileNet\_v2  | 224\*224   | 608M          | ImageNet Train                          | ImageNet Validataion    | 0\.6649/0\.872362              | 0\.635219/0\.850701           |
| 6    | Image Classification     | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | tensorflow | resnet50       | 224\*224   | 6\.97G        | ImageNet Train                          | ImageNet Validataion    | 0\.7520/0\.9219                | 0\.7420/0\.9209               |
| 7    | Image Classification     | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | tensorflow | inception\_v1  | 224\*224   | 3\.0G         | ImageNet Train                          | ImageNet Validataion    | 0\.6976/0\.8963                | 0\.6786/0\.8885               |
| 8    | Image Classification     | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | tensorflow | MobileNet\_v2  | 224\*224   | 1\.17G        | ImageNet Train                          | ImageNet Validataion    | 0\.7487/0\.9250                | 0\.2720/\-                    |
| 9    | ADAS Vehicle Detection   | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | caffe      | VGG\-16        | 360\*480   | 6\.3G         | bdd100k \+ private data                 | bdd100k \+ private data | 0\.426                         | 0\.424                        |
| 10   | ADAS Pedstrain Detection | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | caffe      | VGG\-bn\-16    | 360\*640   | 5\.9G         | coco2014\_train\_person and crowndhuman | coco2014\_val\_person   | 0\.5899                        | 0\.585                        |
| 11   | Traffic Detection        | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | caffe      | VGG\-16        | 360\*480   | 11\.6G        | private data                            | private data            | 0\.602                         | 0\.588                        |
| 12   | Object Detection         | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | caffe      | MobileNet\_v2  | 360\*480   | 6\.57G        | bdd100k train                           | bdd100k val             | 0\.3186                        | 0\.3019                        |
| 13   | Object Detection         | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | tensorflow | VGG\-bn\-16    | 300\*300   | 64\.81G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7942\(11 points\)           | 0\.7882\(11 points\)          |
| 14   | Face Detection           | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | caffe      | VGG\-16        | 320\*320   | 0\.49G        | wider\_face                             | FDDB                    | 0\.8818                        | 0\.8768                       |
| 15   | Face Detection           | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | caffe      | VGG\-16        | 360\*640   | 1\.11G        | wider\_face                             | FDDB                    | 0\.8909                        | 0\.8909                       |
| 16   | ADAS Detection           | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | darknet    | darknet\-53    | 256\*512   | 5\.46G        | cityscape train                         | cityscape val           | 55\.20%                        | 53\.00%                       |
| 17   | Object Detection         | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | darknet    | darknet\-53    | 416\*416   | 65\.42G       | voc07\+12\_trainval                     | voc07\_test             | 82\.4%\(MaxIntegral\)          | 81\.5%\(MaxIntegral\)         |
| 18   | Object Detection         | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | tensorflow | darknet\-53    | 416\*416   | 65\.63G       | voc07\+12\_trainval                     | voc07\_test             | 78\.46%\(11 points\)           | 77\.38%\(11 points\)          |
| 19   | Object Detection         | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | caffe      | VGG\-bn\-16    | 360\*480   | 25G           | coco2014\_train\_person                 | coco2014\_val\_person   | 67\.68%                        | 67\.47%                       |
| 20   | Object Detection         | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | caffe      | VGG\-bn\-16    | 360\*480   | 10\.10G       | coco2014\_train\_person                 | coco2014\_val\_person   | 64\.60%                        | 64\.50%                       |
| 21   | Object Detection         | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | caffe      | VGG\-bn\-16    | 360\*480   | 5\.08G        | coco2014\_train\_person                 | coco2014\_val\_person   | 60\.89%                        | 60\.65%                       |
| 22   | ADAS Segmentation        | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | caffe      | Google\_v1\_BN | 256\*512   | 8\.9G         | Cityscapes gtFineTrain\(2975\)          | Cityscapes Val\(500\)   | 0\.5669                        | 0\.5645                       |
| 23   | ADAS Lane Detection      | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | caffe      | VGG            | 480\*640   | 2\.5G         | caltech\-lanes\-train\-dataset          | caltech lane            | 88\.639%\(F1\-score\)          | 87%\(F1\-score\)              |
| 24   | Pose Estimation          | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | caffe      | Google\_v1\_BN | 128\*224   | 548\.6M       | ai\_challenger                          | ai\_challenger          | 88\.2%\(PCKh0\.5\)             | 87\.86%\(PCKh0\.5\)           |
| 25   | Pose Estimation          | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | caffe      | VGG            | 368\*368   | 49\.88G       | ai\_challenger                          | ai\_challenger          | 0\.45067\(OKs\)                | 0\.44287\(Oks\)               |
| 26   | Object Detection         | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | darknet    | darknet\-19    | 448\*448   | 34G           | voc07\+12\_trainval                     | voc07\_test             | 78\.45%\(MaxIntegral\)         | 77\.39%\(MaxIntegral\)        |
| 27   | Object Detection         | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | darknet    | darknet\-19    | 448\*448   | 11\.56G       | voc07\+12\_trainval                     | voc07\_test             | 77%\(MaxIntegral\)             | 76%\(MaxIntegral\)            |
| 28   | Object Detection         | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | darknet    | darknet\-19    | 448\*448   | 9\.86G        | voc07\+12\_trainval                     | voc07\_test             | 76\.7%\(MaxIntegral\)          | 75\.3%\(MaxIntegral\)         |
| 29   | Object Detection         | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | darknet    | darknet\-19    | 448\*448   | 7\.82G        | voc07\+12\_trainval                     | voc07\_test             | 75\.76%\(MaxIntegral\)         | 74\.6%\(MaxIntegral\)         |
| 30   | Image Classifiction      | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | caffe      | inception      | 299\*299   | 24\.5G        | ImageNet Train                          | ImageNet Validataion    | 79\.59%/94\.70%                | 78\.99%/94\.45%               |
| 31   | Image Classifiction      | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | caffe      | squeezenet     | 227\*227   | 0\.76G        | ImageNet Train                          | ImageNet Validataion    | 54\.64%/78\.20%                | 50\.69%/77\.01%               |
| 32   | Face Recognition         | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | caffe      | lenet          | 96\*72     | 0\.14G        | celebA                                  | processed helen         | 0\.03704\(MAE\)                | 0\.03692\(MAE\)               |
| 33   | Re\-identification       | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | caffe      | resnet18       | 160\*80    | 0\.95G        | Market1501\+CUHK03                      | Market1501              | 78\.00%                        | 77\.60%                       |
| 34   | Object Detection         | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | caffe      | darknet\-53    | 288\*512   | 53\.7G        | bdd100k                                 | bdd100k                 | 50\.60%                        | 49\.14%                       |
| 35   | Image Classifiction      | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | tensorflow | MobileNet\_v1  | 224\*224   | 1\.14G        | ImageNet Train                          | ImageNet Validataion    | 71\.06%/89\.72%                | 67\.87%/87\.67%               |
| 36   | Image Classifiction      | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | caffe      | resnet18       | 224\*224   | 3\.65G        | ImageNet Train                          | ImageNet Validataion    | 68\.44%/88\.64%                | 66\.94%/88\.25%               |
| 37   | Image Classifiction      | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | tensorflow | resnet18       | 224\*224   | 28G           | ImageNet Train                          | ImageNet Validataion    | 68\.91%/88\.63%                | 69\.86%/88\.96%               |

</details>

### Naming Rules
Model name: `F_M_D_H_W_(P)_C`
* `F` specifies training framework: `cf` is Caffe, `tf` is Tensorflow, `dk` is Darknet, `pt` is PyTorch
* `M` specifies the model
* `D` specifies the dataset
* `H` specifies the height of input data
* `W` specifies the width of input data
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned.
* `C` specifies the computation of the model: how many Gops per image

For example, `cf_refinedet_coco_480_360_0.8_25G` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `480*360`, `80%` pruned, and the computation per image is `25Gops`.



## Model Download
The following table lists various models, download link and MD5 checksum for the zip file of each model.

**Note:** To download all the models, visit [all_models.zip](https://www.xilinx.com/bin/public/openDownload?filename=all_models.zip). 

<details>
 <summary><b>Click here to view details</b></summary>

If you are a:
 - Linux user, use the [`get_model.sh`](reference-files/get_model.sh) script to download all the models.   
 - Windows user, use the download link listed in the following table to download a model.


| No\. | Model                          | Size       | Download link                                                                                                      | Checksum                         |
|------|--------------------------------|------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------|
| 1    | resnet50                       | 226\.61 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet50_imagenet_224_224_7.7G.zip             | a1158f0558254b94bbf05651b04893af |
| 2    | Inception\_v1                  | 86\.47 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv1_imagenet_224_224_3.16G.zip         | 9cad57664719e106d1dfe81f0730e1a2 |
| 3    | Inception\_v2                  | 143\.38 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv2_imagenet_224_224_4G.zip             | 13439f7c01b769f72724d0d9bd5f1f87 |
| 4    | Inception\_v3                  | 212\.43 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv3_imagenet_299_299_11.4G.zip         | f6415422c49087dfbc933fd0d2e451ed |
| 5    | mobileNet\_v2                  | 33\.17 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_mobilenetv2_imagenet_224_224_0.59G.zip         | a698a297abc8607503e15f47ea5de539 |
| 6    | tf\_resnet50                   | 204\.41 MB | https://www.xilinx.com/bin/public/openDownload?filename=tf_resnet50_imagenet_224_224_6.97G.zip            | ffce2c0461d0e914d6d1eb3e81b0c825 |
| 7    | tf\_inception\_v1              | 53\.44 MB  | https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv1_imagenet_224_224_3G.zip             | 64f58dd36e28726a62b964284bb91508 |
| 8    | tf\_mobilenet\_v2              | 49\.84 MB  | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_imagenet_224_224_1.17G.zip         | 47e70eae53af73e77664d9871456511f |
| 9    | ssd\_adas\_pruned\_0\.95       | 10\.97 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdadas_bdd_360_480_0.95_6.3G.zip            | 02c14f5b3a4641bef2f6713625f9bf95 |
| 10   | ssd\_pedestrain\_pruned\_0\.97 | 7\.32 MB   | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdpedestrian_coco_360_640_0.97_5.9G.zip     | d913a529e8885451b670f865bec21c3a |
| 11   | ssd\_traffic\_pruned\_0\.9     | 17\.49 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdtraffic_360_480_0.9_11.6G.zip              | a978c750f14b879c45daf0379198c015 |
| 12   | ssd\_mobilnet\_v2              | 98\.48 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdmobilenetv2_bdd_360_480_6.57G.zip           | bbd9b6a5429db3341115df8eb19d30cc |
| 13   | tf\_ssd\_voc                   | 209\.66 MB | https://www.xilinx.com/bin/public/openDownload?filename=tf_ssd_voc_300_300_64.81G.zip                     | 9f7081ec490148eb4709c0075b6db58e |
| 14   | densebox\_320\_320             | 4\.64 MB   | https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_320_320_0.49G.zip               | e7cf3260a84422640f115e4ae62bd963 |
| 15   | densebox\_360\_640             | 4\.64 MB   | https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_360_640_1.11G.zip               | 53da8c489d73c72ad94b38f624157380 |
| 16   | yolov3\_adas\_prune\_0\.9      | 35\.81 MB  | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_cityscapes_256_512_5.46G.zip            | 20530268484ff9a2ff67804ad1c19b3b |
| 17   | yolov3\_voc                    | 940\.03 MB | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_voc_416_416_65.42G.zip                  | d8265f80521da8e3251ea57798818c31 |
| 18   | tf\_yolov3\_voc                | 500\.07 MB | https://www.xilinx.com/bin/public/openDownload?filename=tf_yolov3_voc_416_416_65.63G.zip                  | c5923313c7570226d4a9249ea68b6fdd |
| 19   | refinedet\_pruned\_0\.8        | 10\.2 MB   | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_360_480_0.92_10.10G.zip       | b3fa2804b699915e3dc6bf88478308d8 |
| 20   | refinedet\_pruned\_0\.92       | 5\.07 MB   | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_360_480_0.96_5.08G.zip        | 51e8fb7639786a476829c8286b7e1843 |
| 21   | refinedet\_pruned\_0\.96       | 37\.34 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_360_480_0.8_25G.zip            | 8ae8521ad5d754bb473a2527dfa5a805 |
| 22   | FPN                            | 55\.98 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_fpn_cityscapes_256_512_8.9G.zip                | 2f29e526a604f81ae07654a5c5f50dc8 |
| 23   | VPGnet\_pruned\_0\.99          | 6\.89 MB   | https://www.xilinx.com/bin/public/openDownload?filename=cf_VPGnet_caltechlane_480_640_0.99_2.5G.zip     | 697672ac6d91418e16c19978889cb827 |
| 24   | SP\-net                        | 17\.32 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_SPnet_aichallenger_224_128_0.54G.zip           | 41769a269984a183362f2492f719a0d1 |
| 25   | Openpose\_pruned\_0\.3         | 315\.37 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_openpose_aichallenger_368_368_0.3_189.7G.zip | 3e2f9fac5dcdfbc30d663b2f218ebc6c |
| 26   | yolov2\_voc                    | 476\.34 MB | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_34G.zip                      | a6f439314bdf65d0d4684c8cdc96c3dd |
| 27   | yolov2\_voc\_pruned\_0\.66     | 223\.22 MB | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.66_11.56G.zip           | 9fa27b6cfe81e5f3a62004dc12cabbe7 |
| 28   | yolov2\_voc\_pruned\_0\.71     | 202\.25 MB | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.71_9.86G.zip            | 6a67d3182cf52dae2023ef3255c128e6 |
| 29   | yolov2\_voc\_pruned\_0\.77     | 146\.51 MB | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.77_7.82G.zip            | 662857523d9762c7fe74cc3597cf5fd6 |
| 30   | Inception\-v4                  | 380\.38 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv4_imagenet_299_299_24.5G.zip         | e75b600ca020446626b6700b04ba5f5f |
| 31   | SqueezeNet                     | 11\.27 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_squeeze_imagenet_227_227_0.76G.zip             | 20befe2e854d1e36230e77f283ee3d39 |
| 32   | face\_landmark                 | 50\.42 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_landmark_celeba_96_72_0.14G.zip                | 44236176d313f8a51098d060cf3ad07d |
| 33   | reid                           | 98\.33 MB  | https://www.xilinx.com/bin/public/openDownload?filename=cf_reid_marketcuhk_160_80_0.95G.zip               | bb2ca45bf1e57949a66cb3bf52adce8f |
| 34   | yolov3\_bdd                    | 944\.14 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_yolov3_bdd_288_512_53.7G.zip                   | 25802e6b0e0ae0ac3f0ccea105d2a829 |
| 35   | tf\_mobilenet\_v1              | 42\.43 MB  | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_imagenet_224_224_1.14G.zip         | 4337b02322441ce1686ce19fc1a36d82 |
| 36   | resnet18                       | 178\.45 MB | https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet18_imagenet_224_224_3.65G.zip            | 2380212df49e7c9584bdaef646c470f7 |
| 37   | resnet18\_wide                 | 393\.64 MB | https://www.xilinx.com/bin/public/openDownload?filename=tf_resnet18_imagenet_224_224_28G.zip               | 32f782a084f2f2de089c9eb4f1c3e364 |
| /    | All models                     | 6\.31GB    | https://www.xilinx.com/bin/public/openDownload?filename=all_models.zip                                         | 0fc242102699cad110027ecfff453d91 |

</details>

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, their download link and MD5 checksum for the zip file of each model, see [Model Download](#model-download).

#### Caffe Model Directory Structure
For a Caffe model, you should see the following directory structure:
    
    ├── labelmap.prototxt               # Contains information of the detection class for some models 
    │                                     such as SSD, RefineDet.
    ├── readme.md                       # Contains the environment requirement and data preprocess information. 
    │                                     Refer this file to know more about creating `float.prototxt` by adding
    │                                     datalayer to `test.prototxt` in the `float` directory.
    ├── deploy                          
    │   ├── deploy.caffemodel           # Input to the compiler. The same with deploy.caffemodel in the `fix` directory.
    │   └── deploy.prototxt             # Input to the compiler. The modified prototxt based on deploy.prototxt
    │                                     in the `fix` directory, which removes unnecessary or unsupported layers 
    │                                     for compilation.
    ├── fix                             
    │   ├── deploy.caffemodel           # Quantized weights, the output of decent_q without modification.
    │   ├── deploy.prototxt             # Quantized prototxt, the output of decent_q without modification.
    │   ├── fix_test.prototxt           # Used to run evaluation with fix_train_test.caffemodel on GPU 
    │   │                                 using python test code released in near future. Some models 
    │   │                                 don't have this file if they are converted from Darknet (Yolov2, Yolov3),
    │   │                                 Pytorch (ReID) or there is no Caffe Test (Densebox).
    │   ├── fix_train_test.caffemodel   # Quantized weights can be used for fixed-point training and evaluation.    
    │   └── fix_train_test.prototxt     # Used for fixed-point training and testing with fix_train_test.caffemodel
    │                                     on GPU when datalayer modified to user's data path.
    └── float                           
        ├── float.caffemodel            # Trained float-point weights.
        ├── float.prototxt              # Modified test.prototxt as the input to decent_q along 
        │                                 with float.caffemodel. decent_q is Xilinx quantization tool 
        │                                 which quantizes float-point to fixed-point model with minimal 
        │                                 accuracy loss. 
        ├── test.prototxt               # Used to run evaluation with python test codes released in near future.    
        └── trainval.prorotxt           # Used for training and testing with caffe train/test command 
                                          when datalayer modified to user's data path. Some models don't 
                                          have this file if they are converted from Darknet (Yolov2, Yolov3),
                                          Pytorch (ReID) or there is no Caffe Test (Densebox).          
     

**Note:** For more information on `decent_q`, see the [DNNDK User Guide](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf).

#### Tensorflow Model Directory Structure
For a Tensorflow model, you should see the following directory structure:

    
    ├── input_fn.py                     # Python function to read images in calibration dataset and do data preprocess.
    ├── readme.md                       # Contains the environment requirement, the input and output nodes as well as 
    │                                     the data preprocess and postprocess information.
    ├── fix                          
    │   ├── deploy.model.pb             # Quantized model for the compiler (extended Tensorflow format).
    │   └── quantize_eval_model.pb      # Quantized model for evaluation.
    └── float                             
        └── frozen.pb                   # Float-point frozen model, the input to the `decent_q`.



## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [DNNDK™ (Deep Neural Network Development Kit)](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge) and [Xilinx AI SDK](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections. 

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/support/documentation/ip_documentation/dpu/v3_0/pg338-dpu.pdf).


**Note:** The model performance number listed in the following sections is generated with DNNDK v3.1 and Xilinx AI SDK v2.0.x. For each board, a different DPU configuration is used. DNNDK and Xilinx AI SDK can be downloaded for free from [https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge).

### Performance on ZCU102 (0432055-04)
<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-04)` board with a  `3 * B4096  @ 287MHz   V1.4.0` DPU configuration:


| No\. | Model                          | Name                                                | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
|------|--------------------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|----------------------------------------|
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 12\.85                          | 77\.8                                   | 179\.3                                 |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 5\.47                           | 182\.683                                | 485\.533                               |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 6\.76                           | 147\.933                                | 373\.267                               |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 17                              | 58\.8333                                | 155\.4                                 |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 4\.09                           | 244\.617                                | 638\.067                               |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 11\.94                          | 83\.7833                                | 191\.417                               |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 6\.72                           | 148\.867                                | 358\.283                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 5\.46                           | 183\.117                                | 458\.65                                |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 11\.33                          | 88\.2667                                | 320\.5                                 |
| 10   | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 12\.96                          | 77\.1833                                | 314\.717                               |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 17\.49                          | 57\.1833                                | 218\.183                               |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 24\.21                          | 41\.3                                   | 141\.233                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 69\.28                          | 14\.4333                                | 46\.7833                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 2\.43                           | 412\.183                                | 1416\.63                               |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 5\.01                           | 199\.717                                | 719\.75                                |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 11\.09                          | 90\.1667                                | 259\.65                                |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 70\.51                          | 14\.1833                                | 44\.4                                  |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 70\.75                          | 14\.1333                                | 44\.0167                               |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 29\.91                          | 33\.4333                                | 109\.067                               |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 15\.39                          | 64\.9667                                | 216\.317                               |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 11\.04                          | 90\.5833                                | 312                                    |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 16\.58                          | 60\.3                                   | 203\.867                               |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 9\.44                           | 105\.9                                  | 424\.667                               |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 1\.73                           | 579\.067                                | 1620\.67                               |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 279\.07                         | 3\.58333                                | 16\.55                                 |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 39\.76                          | 25\.15                                  | 86\.35                                 |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 18\.42                          | 54\.2833                                | 211\.217                               |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 16\.42                          | 60\.9167                                | 242\.433                               |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 14\.46                          | 69\.1667                                | 286\.733                               |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 34\.25                          | 29\.2                                   | 84\.25                                 |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 3\.6                            | 277\.65                                 | 1080\.77                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 1\.13                           | 885\.033                                | 1623\.3                                |
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 2\.67                           | 375                                     | 773\.533                               |
| 34   | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | 73\.89                          | 13\.5333                                | 42\.8833                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 3\.2                            | 312\.067                                | 875\.967                               |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 5\.1                            | 195\.95                                 | 524\.433                               |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 33\.28                          | 30\.05                                  | 83\.4167                               |
</details>


### Performance on ZCU102 (0432055-05)   
<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 287MHz   V1.4.0` DPU configuration: 


| No\. | Model                          | Name                                                | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
|------|--------------------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|----------------------------------------|
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 12\.98                          | 77\.0167                                | 163\.417                               |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 5\.51                           | 181\.65                                 | 452\.4                                 |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 6\.8                            | 147                                     | 345\.7                                 |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 17\.11                          | 58\.45                                  | 144\.9                                 |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 4\.13                           | 241\.9                                  | 587\.25                                |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 12\.07                          | 82\.85                                  | 173\.267                               |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 6\.77                           | 147\.65                                 | 330\.583                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 5\.52                           | 181\.067                                | 422\.15                                |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 11\.32                          | 88\.3167                                | 306\.267                               |
| 10   | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 12\.96                          | 77\.1667                                | 309\.4                                 |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 17\.48                          | 57\.2                                   | 216                                    |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 24\.67                          | 40\.5333                                | 124\.733                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 69\.61                          | 14\.3667                                | 46\.9833                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 2\.46                           | 406\.2                                  | 1311\.8                                |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 5\.04                           | 198\.533                                | 645\.567                               |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 11\.16                          | 89\.6333                                | 239\.667                               |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 70\.67                          | 14\.15                                  | 43\.6167                               |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 71\.01                          | 14\.0833                                | 43\.0833                               |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 29\.94                          | 33\.4                                   | 107\.533                               |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 15\.48                          | 64\.6167                                | 210\.817                               |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 11\.06                          | 90\.45                                  | 298\.217                               |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 16\.68                          | 59\.95                                  | 188\.533                               |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 9\.39                           | 106\.45                                 | 396\.85                                |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 1\.74                           | 574\.833                                | 1516\.78                               |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 279\.07                         | 3\.58333                                | 16\.6333                               |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 39\.84                          | 25\.1                                   | 84\.5667                               |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 18\.44                          | 54\.2333                                | 206\.067                               |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 16\.44                          | 60\.8167                                | 238\.017                               |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 14\.48                          | 69\.0667                                | 279\.35                                |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 34\.46                          | 29\.0167                                | 78\.5                                  |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 3\.64                           | 274\.767                                | 1012\.17                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 1\.15                           | 871\.333                                | 1444\.25                               |
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 2\.7                            | 370\.317                                | 702\.8                                 |
| 34   | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | 74\.07                          | 13\.5                                   | 42\.0833                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 3\.23                           | 309\.65                                 | 809\.5                                 |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 5\.18                           | 193\.067                                | 477\.05                                |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 33\.41                          | 29\.9333                                | 80\.0667                               |

</details>

### Performance on FPGA board: ZCU104  
<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 305MHz   V1.4.0` DPU configuration: 


| No\. | Model                          | Name                                                | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
|------|--------------------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|----------------------------------------|
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 12\.13                          | 82\.45                                  | 151\.8                                 |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 5\.07                           | 197\.333                                | 404\.933                               |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 6\.33                           | 158\.033                                | 310\.15                                |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 16\.03                          | 62\.3667                                | 126\.283                               |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 3\.85                           | 259\.833                                | 536\.95                                |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 11\.31                          | 88\.45                                  | 163\.65                                |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 6\.35                           | 157\.367                                | 305\.467                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 5\.21                           | 191\.867                                | 380\.933                               |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 10\.69                          | 93\.5333                                | 242\.917                               |
| 10   | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 12\.13                          | 82\.45                                  | 236\.083                               |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 16\.48                          | 60\.6667                                | 159\.617                               |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 37\.78                          | 26\.4667                                | 116\.433                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 75\.09                          | 13\.3167                                | 33\.5667                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 2\.33                           | 428\.533                                | 1167\.35                               |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 4\.65                           | 215\.017                                | 626\.317                               |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 10\.51                          | 95\.1667                                | 228\.383                               |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 66\.37                          | 15\.0667                                | 33                                     |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 66\.74                          | 14\.9833                                | 32\.8                                  |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 28                              | 35\.7167                                | 79\.1333                               |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 14\.54                          | 68\.7833                                | 160\.6                                 |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 10\.39                          | 96\.2333                                | 241\.783                               |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 15\.72                          | 63\.6167                                | 177\.333                               |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 8\.91                           | 112\.233                                | 355\.717                               |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 1\.6                            | 626\.5                                  | 1337\.33                               |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 267\.86                         | 3\.73333                                | 12\.1333                               |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 37\.66                          | 26\.55                                  | 63\.7833                               |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 17\.51                          | 57\.1167                                | 158\.917                               |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 15\.63                          | 63\.9667                                | 186\.867                               |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 13\.78                          | 72\.55                                  | 224\.883                               |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 32\.33                          | 30\.9333                                | 64\.6                                  |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 3\.52                           | 284\.033                                | 940\.917                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 1\.02                           | 977\.683                                | 1428\.2                                |
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 2\.45                           | 407\.583                                | 702\.717                               |
| 34   | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | 69\.77                          | 14\.3333                                | 31\.7                                  |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 3\.03                           | 330\.25                                 | 728\.35                                |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 4\.84                           | 206\.65                                 | 428\.55                                |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 31\.23                          | 32\.0167                                | 62\.7667                               |

</details>

### Performance on Ultra96
<details>
 <summary><b>Click here to view details</b></summary>
 
The following table lists the performance number including end-to-end throughput and latency for each model on the `Ultra96` board with a `1 * B1600  @ 287MHz   V1.4.0` DPU configuration: 

**Note:** The original power supply of Ultra96 is not designed for high performance AI workload. The board may occasionally hang to run few models, When multi-thread is used. For such situations, `NA` is specified in the following table.


| No\. | Model                          | Name                                                | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
|------|--------------------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|----------------------------------------|
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 30\.8                           | 32\.4667                                | 33\.4667                               |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 13\.98                          | 71\.55                                  | 75\.0667                               |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 17\.16                          | 58\.2667                                | 61\.2833                               |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 44\.05                          | 22\.7                                   | 23\.4333                               |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 7\.34                           | 136\.183                                | NA                                     |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 28\.02                          | 35\.6833                                | 36\.6                                  |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 16\.96                          | 58\.9667                                | 61\.2833                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 10\.17                          | 98\.3                                   | 104\.25                                |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 24\.3                           | 41\.15                                  | 46\.2                                  |
| 10   | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 23\.29                          | 42\.9333                                | 50\.8                                  |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 35\.5                           | 28\.1667                                | 31\.8                                  |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 60\.79                          | 16\.45                                  | 27\.8167                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 186\.92                         | 5\.35                                   | 5\.81667                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 4\.17                           | 239\.883                                | 334\.167                               |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 8\.55                           | 117                                     | 167\.2                                 |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 22\.79                          | 43\.8833                                | 49\.6833                               |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 185\.19                         | 5\.4                                    | 5\.53                                  |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 199\.34                         | 5\.01667                                | 5\.1                                   |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 66\.37                          | 15\.0667                                | NA                                     |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 32\.17                          | 31\.0883                                | 33\.6667                               |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 20\.29                          | 49\.2833                                | 55\.25                                 |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 36\.34                          | 27\.5167                                | NA                                     |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 13\.9                           | 71\.9333                                | NA                                     |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 3\.82                           | 261\.55                                 | 277\.4                                 |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 560\.75                         | 1\.78333                                | NA                                     |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 118\.11                         | 8\.46667                                | 8\.9                                   |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 37\.5                           | 26\.6667                                | 30\.65                                 |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 30\.99                          | 32\.2667                                | 38\.35                                 |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 26\.29                          | 38\.03333                               | 46\.8333                               |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 88\.76                          | 11\.2667                                | 11\.5333                               |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 5\.96                           | 167\.867                                | 283\.583                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 2\.95                           | 339\.183                                | 347\.633                               |
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 6\.28                           | 159\.15                                 | 166\.633                               |
| 34   | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | 193\.55                         | 5\.16667                                | 5\.31667                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 5\.97                           | 167\.567                                | 186\.55                                |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 13\.47                          | 74\.2167                                | 77\.8167                               |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 97\.72                          | 10\.2333                                | 10\.3833                               |
</details>

## Contributing 
We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

* [GitHub Issues](https://github.com/Xilinx/TechDocs/issues)
* [Forum](https://forums.xilinx.com/t5/Deephi-DNNDK/bd-p/Deephi)
* <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.

## License

Xilinx AI Model Zoo is licensed under [Apache License Version 2.0](reference-files/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<hr/>
<p align="center"><sup>Copyright&copy; 2019 Xilinx</sup></p>
