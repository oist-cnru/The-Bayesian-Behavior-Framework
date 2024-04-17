function  plot_adaptation_readaptation_progress(details_dir)


id = 2; % you may change to check different random seeds

N = 5; 

episodes = [0, 5000, 15000, 20000, 25000, 27500, 30000, 37000];

% ----------------------------------------------

figure;
set(gcf, 'Position', [200, 400, 1660, 230]);

count = 0;
for stp = episodes 
    
    % draw the maze shape
    count = count + 1;
    subplot(2, length(episodes) / 2, count)
    
    draw_tmaze;
    
    for k = 0: 500 : 500 * (N - 1)
        
        filename = sprintf("tmaze_habitization_%d_episode_%d.mat", id, stp+k);

        data = load(strcat(details_dir,  filename));

        hold on
        e_len = size(data.info,1);

        x_traj = data.info(1:e_len,1);
        y_traj = data.info(1:e_len,2);


        quiverDQ(x_traj, y_traj, 0.4, 0.8, hsv2rgb([0.4, 0.0, 0.7]), 0.5)

    end


end


end

