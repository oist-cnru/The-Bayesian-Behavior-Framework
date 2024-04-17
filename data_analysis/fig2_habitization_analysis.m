function fig2_habitization_analysis(dataPath)

    line_styles = {'-', '-.', '--', ':', '-'};
    
    plot_colors = {hsv2rgb([0.55, 0.95, 0.95]), ...
                            hsv2rgb([0.88, 0.8, 0.8]), ...
                            hsv2rgb([0.1, 0.65, 0.65]), ...
                            hsv2rgb([0.29, 0.5, 0.5]), ...
                            hsv2rgb([0.66, 0.35, 0.35])};
    
    figure('Position', [100, 100, 1800, 900])
    
    env_name = "tmaze";
    max_all_steps = 650000;
    
    
    avg_perf = [];
    ids = [];
    episode_rewards = [];
    sigma_prior = [];
    sigma_post = [];
    kld = [];
    returns = [];
    habitual_ratio = [];
    aif_iterations = [];
    steps_taken = [];
    recon_loss = [];
    loss_rl = [];
    loss_a = [];
    loss_q = [];
    loss_v = [];
    
    count = 0 ;
    listing = dir(dataPath);
    
    for i = 3:length(listing)
        name = listing(i).name;
    
        if ~contains(name, "habitization")
            continue
        end
    
        data = load(strcat(dataPath, name));
    
        count = count + 1;
    
        sig_priors = reshape(mean(double(data.sig_priors), 3), 1, []);
        sig_priors = sig_priors(100001:max_all_steps);
        
        sigma_prior(:, count) = reshape(smooth(sig_priors, 1000), 1, []);
    
        sig_posts =  reshape(mean(double(data.sig_posts), 3), 1, []);
        sig_posts = sig_posts(100001:max_all_steps);
        sigma_post(:, count) = reshape(smooth(sig_posts, 1000), 1, []);
    
        returns(:, count) = reshape(data.performance_wrt_step, 1, []);
        steps_taken(:, count) = reshape(data.steps_taken_wrt_step, 1, []);
    
        habitual_ratio(:, count)  = reshape(smooth(sig_priors.^(-2) ./ (sig_posts.^(-2) + sig_priors.^(-2)), 1000), 1, []);
        aif_iterations(:, count)  = reshape(smooth(double(data.aif_iterations(100001:max_all_steps)), 1000), 1, []);
    
        kld(:, count)  = reshape(data.kld_all, 1, []);
        recon_loss(:, count)  = reshape(-data.logp_x_all, 1, []);
        loss_a(:, count) = reshape(data.loss_a_all * 1e5, 1, []);
        loss_q(:, count) = reshape(data.loss_q_all, 1, []);
        loss_v(:, count) = reshape(data.loss_v_all, 1, []);
    
    end
    
    % ------------- plot ---------------------
    for data_type = 1: 6
            
        hs = [];
       
        if data_type == 1
            subplot(4, 2, 1) 
            plot_interval = 1;
            smooth_steps = 3;
            data_to_plot_list = {returns(1:plot_interval:end, :)};
            ylabel("rewards")
            ylim([-50, 1200])
    
        elseif data_type == 3
            subplot(4, 2, 3)  %left
            plot_interval = 1;
            smooth_steps = 3;
            data_to_plot_list = {steps_taken(1:plot_interval:end, :)};
            ylabel("steps")
            ylim([-50, 1200])
    
        elseif data_type == 5
            subplot(2, 2, 3)  %left
            plot_interval = 20;
            smooth_steps = 20;
            data_to_plot_list = {recon_loss(1:plot_interval:end, :), ...
                                            kld(1:plot_interval:end,:), ...
    %                                         loss_a(1:plot_interval:end,:), ...
    %                                         loss_v(1:plot_interval:end,:) + loss_q(1:plot_interval:end,:), ...
    %                                         loss_v(1:plot_interval:end,:),...
                                            };
            ylabel("loss (rescaled)")
            yticklabels("")
    
        elseif data_type == 2
            subplot(2, 2, 2)  %right
            plot_interval = 30;
            smooth_steps = 30;
            data_to_plot_list = {sigma_post(1:plot_interval:end, :), ...
                                            sigma_prior(1:plot_interval:end, :)};
            ylabel("STD of intention")
    
        elseif data_type == 4
            subplot(4, 2, 6)  %right
            plot_interval = 30;
            smooth_steps = 30;
            data_to_plot_list = {habitual_ratio(1:plot_interval:end, :)};
            ylabel("habitual ratio")
    
        elseif data_type == 6
            subplot(4, 2, 8)  %right
            plot_interval = 30;
            smooth_steps = 30;
            data_to_plot_list = {aif_iterations(1:plot_interval:end, :)};
            ylabel("iterations in AIf")
        end
                
        box off
        grid on
        hold on
    
        for  i = 1 : length(data_to_plot_list)
            data_to_plot = data_to_plot_list{i};
    
            plot_color = plot_colors{i};
            line_style = line_styles{i};
        
            data_to_plot = movmean(data_to_plot, smooth_steps, 1,  'Endpoints', 'shrink');
    
            mean_value = mean(data_to_plot, 2);
    
            n_seeds = size(data_to_plot, 2);
            
            tx = 1;  
            sem = std(data_to_plot, [], 2) / sqrt(size(data_to_plot, 2)) * tx;  % Standard error of the mean
    
            if data_type == 5
                mean_value = mean_value - min(mean_value);
                scale = max(mean_value);
                mean_value = mean_value / scale;
                sem = sem / scale;
            end
    
            xx = linspace(100001, max_all_steps, length(data_to_plot));
            xx = xx / 1000; % thousand steps
            
            ciplot(mean_value -  sem, mean_value + sem, xx , plot_color)
            h = plot(xx, mean_value, 'color', plot_color, 'linestyle', line_style);    
    
            hs = [hs, h];
        end
        
        xlim([100000/1000, max_all_steps/1000])
    
        ylim auto
    
        if data_type == 2
            if length(hs) == 2
                legend(hs, ["posterior (goal-directed)", "prior (habitual)"]);
            end
        end
    
        if data_type == 5
            ylim([-0.1, 1.95])
            if length(hs) == 2
                legend(hs, ["prediction loss", "KL-divergence"]);
            end
        end
    
        if data_type >= 5
            xlabel('thousand steps')
        end
    
    end
    
    set(gcf, 'color', 'white')

end