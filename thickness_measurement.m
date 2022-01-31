[a, map]=imread('Fig4a_thickness.png');
a = ind2rgb(a,map);
bw=im2bw(a, 0.1);
skeleton=imread('Fig4a_skeleton.png');
skeleton=im2bw(skeleton, 0.5);
branchpoints=bwmorph(skeleton, 'branchpoints');
se=strel('disk', 3);
branchpoints=imdilate(branchpoints,se);
segments=skeleton>branchpoints;
segments=bwareaopen(segments, 50);
stats = regionprops(segments,'Centroid', 'PixelIdxList');
centroids = cat(1, stats.Centroid);
D = bwdist(~bw);
hold on
imagesc(D.*single(skeleton))
imshow(a)
for i=1:numel(stats)
m(i)=mean(D(stats(i).PixelIdxList));
text(centroids(i,1),centroids(i,2), num2str(m(i)), 'Color','w');
end