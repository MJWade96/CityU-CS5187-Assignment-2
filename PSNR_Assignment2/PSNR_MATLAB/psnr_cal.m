clc;
clear;
test_imgs = ["Art", "Dolls", "Reindeer"];

for index =1:3
    gt_names = ".\gt\"+test_imgs(index)+"\disp1.png";
    gt_img = imread(gt_names);
    pgt_img = imread(gt_names);
    
    pred_names =  ".\pred\"+test_imgs(index)+"\disp1.png";
    pred_img = imread(pred_names);
    
 %  When calculating the PSNR:
 % 1.) The pixels in ground-truth disparity map with '0' value will be neglected.
 % 2.) The left part region (1-250 columns) of view1 is not included as there is no
 % corresponding pixels in the view5.
    gt_img = gt_img(:, 251:end);
    pred_img = pred_img(:, 251:end);
    pred_img(gt_img==0)= 0;

    peaksnr = psnr(pred_img,gt_img);
    fprintf('The Peak-SNR value is %0.4f \n', peaksnr);
end