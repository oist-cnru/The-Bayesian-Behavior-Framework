function fig = plot_planning_details(planning_dir)


plot_colors = {hsv2rgb([0.15, 0, 0.3]), ...
                        hsv2rgb([0.3, 0, 0.4]), ...
                        hsv2rgb([0.45, 0, 0.5]), ...
                        hsv2rgb([0.6, 0, 0.6]), ...
                        hsv2rgb([0.75, 0, 0.7]), ...
                        hsv2rgb([0.9, 0, 0.8]), ...
                        };


id = 7;  % can change to see different agents


fig =figure;
set(gcf, 'Position', [200, 200, 1660, 830]);

goals_to_plot =[0, 0, 1, 2, 3, 4] ;

count = 0;

for goal_id = goals_to_plot

    count = count + 1;

    step_to_plot = 5 + count;
    
    filename = sprintf("tmaze_planning_%d_goal_%d_episode_%d.mat", id, goal_id, count);

    data = load(strcat(planning_dir,  filename));

    if goal_id == 1 
        goal_red = data.goal_obs;
    elseif goal_id == 2
        goal_blue = data.goal_obs;
    end

    hold on
    e_len = size(data.info,1);

    x_traj = data.info(1:e_len,1);
    y_traj = data.info(1:e_len,2);

    
    subplot( length(goals_to_plot), 1, count)

    draw_tmaze(0); 

    % plot goal region
       
    if goal_id == 0
        reward_area_range = 1.5;
        gx = data.goal_pos(1) ;
        gy = data.goal_pos(2) ;
        x_tar = [gx +  reward_area_range, gx +  reward_area_range, gx -  reward_area_range, gx -  reward_area_range, gx +  reward_area_range ] ;
        y_tar = [gy -  reward_area_range, gy +  reward_area_range, gy +  reward_area_range, gy -  reward_area_range, gy -  reward_area_range] ;
        
        hold on
        if gx~=0 || gy~=0
            fill(x_tar, y_tar, hsv2rgb([0.4, 0.4, 0.8]), 'edgecolor', 'none')
            plot(x_tar, y_tar, '-', 'color', hsv2rgb([0.4, 0.4, 0.8]))
        end
    
    end

    quiverDQ(x_traj, y_traj, 0.4, 0.8, plot_colors{goal_id+2});
    ylim([-10.5, 6.5])
    xlim([-10.5, 80])
    
   
    imshow1([18, 40] , [-10.5, -5], squeeze(data.obs(step_to_plot, :, :, :)));
       
    if goal_id <= 0
        imshow1([18, 40] , [-1, 4.5], data.goal_obs);
    elseif goal_id == 1 || goal_id == 4
        imshow1([18, 40] , [-1, 4.5], goal_red);
    elseif goal_id == 2 || goal_id == 3
        imshow1([18, 40] , [-1, 4.5], goal_blue);
    end
    
    imshow1([44, 66] , [-10.5, -5], squeeze(data.pred_visions(step_to_plot, 1:3, :, :)));

    imshow1([44, 66] , [-1, 4.5], squeeze(data.pred_visions(step_to_plot, 4:6, :, :)));

%     if count == 1
    text(29, -3, "obs. at step "+num2str(step_to_plot), 'FontName', 'arial', 'HorizontalAlignment', 'center')
    text(29, 6.5, "goal obs.", 'FontName', 'arial', 'HorizontalAlignment', 'center')
    text(55, -3, "pred. obs.", 'FontName', 'arial', 'HorizontalAlignment', 'center')
    text(55, 6.5, "pred. future obs.", 'FontName', 'arial', 'HorizontalAlignment', 'center')
%     end

    plot([data.info(step_to_plot, 1), 18], [data.info(step_to_plot, 2), -10.5], 'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.4)
    plot([data.info(step_to_plot, 1), 18], [data.info(step_to_plot, 2), -5],  'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.4)
    
    
    if goal_id == 0
        plot([data.goal_pos(1), 18], [data.goal_pos(2), -1], 'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.4)
        plot([data.goal_pos(1), 18], [data.goal_pos(2), 4.5],  'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.4)
    end

    if goal_id == goals_to_plot(end - 1) || goal_id == goals_to_plot(end)
        text(29, 1.75,  'avoid', 'FontName', 'arial', 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 15)
    end

end



% ---------------- statistics of planning success rate --------------

fig_planning_success_rate =figure;
set(fig_planning_success_rate, 'Position', [400, 200, 600, 500]);

planning_success_rate = [];
N = 10;

for goal_id = -1:4
    count = 0;


    for id = 0 : 11
        for k = 0 : N-1
            
            filename = sprintf("tmaze_planning_%d_goal_%d_episode_%d.mat", id, goal_id, k);
            data = load(strcat(planning_dir,  filename));

            count = count + 1;
    
            if size(data.info, 1) < 30 && data.reward(end) > 0
                planning_success_rate(count, goal_id + 2) =  1;
            else
                planning_success_rate(count, goal_id + 2) =  0;
            end

        end
    end
end

means = mean(planning_success_rate, 1);
errors = std(planning_success_rate, [], 1) / sqrt(size(planning_success_rate, 1));

means = [means(1), -1, means(2:end)];
errors = [errors(1), 0, errors(2:end)];

p_values = nan(numel(means), numel(means));


for k = 2:6
   %Chi-square test
   % Observed data
   n1 = nnz(planning_success_rate(:,1));
   N1 = length(planning_success_rate(:,1));
   n2 = nnz(planning_success_rate(:,k));
   N2 = length(planning_success_rate(:,k));
   % Pooled estimate of proportion
   p0 = (n1+n2) / (N1+N2);
   % Expected counts under H0 (null hypothesis)
   n10 = N1 * p0;
   n20 = N2 * p0;
   % Chi-square test, by hand
   observed = [n1 N1-n1 n2 N2-n2];
   expected = [n10 N1-n10 n20 N2-n20];
   chi2stat = sum((observed-expected).^2 ./ expected);
   p_values(1,k+1) = 1 - chi2cdf(chi2stat,1);
end


% Make P symmetric, by copying the upper triangle onto the lower triangle
lidx = tril(true(size(p_values)), -1);
PT = p_values';
p_values(lidx) = PT(lidx);

plot([-2, 8], [1, 1], '-', 'color', [0.5, 0.5, 0.5])
hold on

bar_handle = superbar(means, 'E', errors, 'P', p_values, 'PLineOffset', mean(errors)*15, 'PStarFontSize', 14, 'ErrorbarLineWidth', 1.5, 'PLineWidth', 2);
for iBarSeries = 1:6
    k = max(1, iBarSeries - 1);
    set(bar_handle(iBarSeries), 'FaceColor', plot_colors{k}, 'EdgeColor', 'none');
end

% for i = 1:6
%     plot(planning_success_rate(:, i) - planning_success_rate(:, i) + i, planning_success_rate(:, i), 'k.', 'MarkerSize', 5);
% end

% bar_handle = bar(means, 'FaceColor','flat', 'EdgeColor', 'none');
% 
% 
% 
% for k = 1:6
%     bar_handle.CData(k,:) = plot_colors{k};
% end

% Adding error bars
% hold on;
% errorbar(1:6, means, errors, 'LineWidth', 2, 'LineStyle','none', 'Color', [0.4, 0.4, 0.4]);

ylabel('Planning success rate');

xticks([1,3,4,5,6, 7])
xticklabels(["habitual", "full obs.", "see red", "see blue", "avoid blue", "avoid red"])
yticks([0, 0.5, 1])
ylim([0, 2.7])

xlim([0.5, 7.5])

box off
grid off

end


function handle = imshow1(x, y, img)
    handle = image(x, y, permute(squeeze(img(1:3 , :, :)), [2, 3, 1]));
    axis off
end