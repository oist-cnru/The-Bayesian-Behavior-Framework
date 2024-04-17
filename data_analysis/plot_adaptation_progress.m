function fig = plot_adaptation_progress(details_dir)

if ~exist("details_dir", 'var')
    % ours
    details_dir = ['C:\code\amlt\BB_sweep_beta\search', ...
        '_mpz_0.1', ...
        '\details\'];
    explains{1} = "ours";
end


id = 7;

N = 10; 

episodes = [0, 2000, 5000, 10000];

% ----------------------------------------------

fig =figure;
set(gcf, 'Position', [200, 400, 1660, 230]);

count = 0;
for stp = episodes 
    
    % draw the maze shape
    count = count + 1;
    subplot(1, length(episodes), count)
    
    draw_tmaze;
    
    for k = 0: 100 : 100 * (N - 1)
        
        filename = sprintf("tmaze_planning_%d_episode_%d.mat", id, stp+k);

        data = load(strcat(details_dir,  filename));

        hold on
        e_len = size(data.info,1);

        x_traj = data.info(1:e_len,1);
        y_traj = data.info(1:e_len,2);


        quiverDQ(x_traj, y_traj, 0.4, 0.8, hsv2rgb([0.4, 0.0, 0.7]))
%         plot([x_traj(1)], [y_traj(1)], 'o', 'color', hsv2rgb([0.4, 0.0, 0.2]), 'markersize', 5, 'MarkerFaceColor',hsv2rgb([0.4, 0.0, 0.2]))


%         plot(x_traj, y_traj, 'color', [hsv2rgb([0.4, 0.5, 0.7]), 0.66], 'LineWidth', 1.5)

%         plot([x_traj(1)], [y_traj(1)], 'o', 'color', hsv2rgb([0.4, 0.0, 0.6]), 'markersize', 5, 'MarkerFaceColor',hsv2rgb([0.4, 0.0, 0.2]))


    end


%     title(['episode ', num2str(stp)])


end


end

