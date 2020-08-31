clc
clear all
close all

ROC_PathAL = 'Eval_Res/Avenue/frame-level';
All_files = dir([ROC_PathAL, '/*.mat']);
Colors = {'b', 'c', 'k', 'r', 'g', 'm', 'y', };
MarkerTypes = {'o', 'x', '^', 'v', '+', 'p', '<', '>', 'd'};
LineStyle = '-';

for i = 1: length(All_files)
   FilePath = [ROC_PathAL, '/', All_files(i).name];
   load(FilePath)
%    disp(AUC * 100)
   plot(X, Y, MarkerTypes{i}, 'MarkerSize', 6.0, 'LineStyle', LineStyle, 'Color',Colors{i}, 'LineWidth',2.0);
   hold on;
   clear X Y
end

% UMN frame-level
% legend({'Ours', 'Binary classifier', 'Biswas et al.', 'Lu et al.', 'Zhu et al.', 'Li et al.', 'Hasan et al.'}, 'FontSize', 10, 'Location', 'southeast');
% Ped1 frame-level
% legend({'Ours', 'Mehran et al.', 'Mahadevan et al.', 'Xu et al.', 'Lu et al.', 'Adam et al.', 'Zhou et al.'}, 'FontSize', 10, 'Location', 'southeast')
% Ped1 pixel-level
% legend({'Ours', 'Biswas et al.', 'Cheng et al.', 'Lu et al.', 'Zhou et al.', 'Mahadevan et al.', 'Mehran et al.'}, 'FontSize', 10, 'Location', 'southeast')
% Ped2 frame-level
% legend({'Ours', 'Hu et al.', 'Javan et al.', 'Mahadevan et al.', 'Biswas et al.', 'Mehran et al.', 'Xu et al.'}, 'FontSize', 10, 'Location', 'southeast')
% Ped2 pixel-level
% legend({'Ours', 'Biswas et al.', 'Lu et al.', 'Sabokrou et al.', 'Mahadevan et al.', 'Chen et al.', 'Mehran et al.'}, 'FontSize', 10, 'Location', 'southeast')
% Avenue frame-level
legend({'Ours', 'Lu et al.', 'Giorno et al.', 'Hasan et al.', 'Chong et al.'}, 'FontSize', 10, 'Location', 'southeast')
% Avenue pixel-level
% legend({'Ours', 'Lu et al.', 'Giorno et al.', 'Hasan et al.', 'Chong et al.'}, 'FontSize', 10, 'Location', 'southeast')
% LV frame-level
% legend({'Ours', 'Biswas et al.', 'Lu et al.', 'Hasan et al.'}, 'FontSize', 10, 'Location', 'southeast')
% LV ROI-level
% legend({'Ours', 'Biswas et al.', 'Lu et al.'}, 'FontSize', 10, 'Location', 'southeast')

% title('ROC for Abnormal Event Detection')
plot([0,1], [1,0], 'LineStyle','--', 'Color', 'k');
xlabel('False Positive Rate', 'FontWeight', 'normal', 'FontSize',15);
set(gca, 'XTick', 0:0.05:1);
ylabel('True Positive Rate', 'FontWeight', 'normal', 'FontSize',15);
set(gca, 'YTick', 0:0.05:1);

grid on
set(gca, 'GridLineStyle', ':', 'GridAlpha', 1);

set(gca, 'XMinorGrid','off', 'YMinorGrid','off');

a = get(gca,'XTickLabel');  
b = cell(size(a));
b(mod(1:size(b), 4) == 1, :) = a(mod(1:size(a), 4) == 1, :);
set(gca,'XTickLabel',b);

a = get(gca,'YTickLabel');  
b = cell(size(a));
b(mod(1:size(b), 4) == 1, :) = a(mod(1:size(a), 4) == 1, :);
set(gca,'YTickLabel',b);

set(gca, 'FontWeight', 'normal', 'FontSize',10);
