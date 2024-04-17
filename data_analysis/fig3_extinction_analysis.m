function fig3_extinction_analysis(exp_path)

    line_styles = {'-', '-.', '--', ':'};
    
    plot_colors = {hsv2rgb([0.55, 0.95, 0.95]), ...
                            hsv2rgb([0.88, 0.8, 0.8]), ...
                            hsv2rgb([0.1, 0.65, 0.65]), ...
                            hsv2rgb([0.29, 0.5, 0.5])};
    
    
    
    figure('Position', [100, 100, 1800, 900])
    
    smooth_steps = 10;
    
    %%%%%%%%%%%%%%
    
    env_name = "tmaze";
    max_all_steps = 650000;
    
    list = dir(exp_path);
    
    hs = [];
    
    %%%%%%%%%%%%%%%
    ps_str = {};
    dataPaths = {};
    count = 0;
    for i = 3:length(list)
    
        if ~(contains(list(i).name, "180") || contains(list(i).name, "260") || contains(list(i).name, "420"))
            continue
        end
    
    
        count = count + 1;
        ps_str{count} = list(i).name;
    end
    
    explains = ["80k steps in adaptation", ...
                       "160k steps in adaptation", ...
                       "320k steps in adaptation", ...
                       ];
    
    explains_short = ["80k", ...
                                 "160k", ...
                                 "320k", ...
                                   ];
    avg_perf = [];
    ids = [];
    episode_rewards = [];
    sigma_prior = [];
    sigma_post = [];
    kld = [];
    returns = [];
    habitual_ratio = [];
    aif_iterations = [];
    
    stat_data = [];
    
    
    %%%%%%%%%%%%%%%
    
    for hp_id = 1:length(ps_str)
        
        count = 0 ;
        
        dataPaths{hp_id} = strcat(exp_path, ps_str{hp_id}, '\data\');
    
        listing = dir(dataPaths{hp_id});
        
    
        for i = 3:length(listing)
            name = listing(i).name;
    
            if (~contains(name, "habitization") && ~contains(name, "habituation")) || contains(name, ".model")
                continue
            end
    
    
            data = load(strcat(dataPaths{hp_id}, name));
    
            count = count + 1;
    
            sig_priors = reshape(mean(double(data.sig_priors), 3), 1, []);
            sig_priors = sig_priors(100001:max_all_steps);
            
            sigma_prior(:, count, hp_id) = reshape(smooth(sig_priors, 1000), 1, []);
    
            sig_posts =  reshape(mean(double(data.sig_posts), 3), 1, []);
            sig_posts = sig_posts(100001:max_all_steps);
            sigma_post(:, count, hp_id) = reshape(smooth(sig_posts, 1000), 1, []);
    
    
            returns(:, count, hp_id) = reshape(data.performance_wrt_step, 1, []);
            habitual_ratio(:, count, hp_id)  = reshape(smooth(sig_priors.^(-2) ./ (sig_posts.^(-2) + sig_priors.^(-2)), 1000), 1, []);
            aif_iterations(:, count, hp_id)  = reshape(smooth(double(data.aif_iterations(100001:max_all_steps)), 1000), 1, []);
    
            kld(:, count, hp_id)  = reshape(data.kld_all, 1, []);
            fe(:, count, hp_id)  = 0.1 * reshape(data.kld_all, 1, []) - reshape(data.logp_x_all, 1, []);
    
        end
    end
    
    N_data_type = 4;
    thousand_steps_to_consider = 100;
    
    % ------------- plot ---------------------
    for data_type = 1: N_data_type
            
        subplot(N_data_type, 9, [9 * data_type - 8, 9 * data_type - 4])  %left
        
        means = [];
        errors = [];
    
        for hp_id = 1:length(ps_str)
            plot_color = plot_colors{hp_id};
            line_style = line_styles{hp_id};
            
            readapt_start = 100 + str2num(explains_short{hp_id}(1:end-1));
    
            if data_type == 1
                plot_interval = 1;
                data_to_plot = returns(:, :, hp_id);
                ylabel("rewards")
                xx = linspace(100001, max_all_steps, length(data_to_plot));
                xx = xx / 1000; % thousand steps
    
                % statistics
                tmp = zeros([1, size(data_to_plot, 2)]);
                for j = 1:size(data_to_plot, 2)
                    smoothed = movmean(data_to_plot(:, j), smooth_steps, 1,  'Endpoints', 'shrink');
                    tmp(j) =  min(xx(find((smoothed' >  400).*(xx>readapt_start+1)))) - readapt_start;  
                    % steps taken to finish re-adaptation (refractory period)
                end
                
    
    
            elseif data_type == 2
                plot_interval = 30;
                data_to_plot = sigma_prior(:, :, hp_id);
                ylabel("STD of prior intention")
                xx = linspace(100001, max_all_steps, length(data_to_plot));
                xx = xx / 1000; % thousand steps
    
                % statistics
                tmp = zeros([1, size(data_to_plot, 2)]);
                for j = 1:size(data_to_plot, 2)
                    smoothed = movmean(data_to_plot(:, j), smooth_steps, 1,  'Endpoints', 'shrink');
                    tmp(j) = max(smoothed(find((readapt_start + thousand_steps_to_consider >= xx) .* (xx> readapt_start))));  
                end
                tmp = log(tmp);
    
            elseif data_type == 3
                plot_interval = 30;
                data_to_plot = habitual_ratio(:, :, hp_id);
                ylabel("habitual ratio")
                xx = linspace(100001, max_all_steps, length(data_to_plot));
                xx = xx / 1000; % thousand steps
    
                % statistics
                tmp = zeros([1, size(data_to_plot, 2)]);
                for j = 1:size(data_to_plot, 2)
    
                    smoothed = movmean(data_to_plot(:, j), smooth_steps, 1,  'Endpoints', 'shrink');
                    tmp(j) =  min(smoothed(find((readapt_start + thousand_steps_to_consider  >= xx) .* (xx> readapt_start))));  
                end
    
    
            elseif data_type == 4
                plot_interval = 30;
                data_to_plot =fe(:, :, hp_id);
                ylabel("free energy loss")
                xx = linspace(100001, max_all_steps, length(data_to_plot));
                xx = xx / 1000; % thousand steps
    
                % statistics
                tmp = zeros([1, size(data_to_plot, 2)]);
                for j = 1:size(data_to_plot, 2)
    
                    tmp(j) =  mean(data_to_plot(find((readapt_start + thousand_steps_to_consider  >= xx) .* (xx> readapt_start)), j));  
                end
    
            end
    
            % scalar statistics for the barplot
            means(hp_id) = mean(tmp);
            errors(hp_id) =  std(tmp, [], 2) / sqrt(size(tmp, 2));  % Standard error of the mean
            stat_data(:, hp_id, data_type) = tmp;
    
            % result curves
            data_to_plot = movmean(data_to_plot(1:plot_interval:end, :), smooth_steps, 1,  'Endpoints', 'shrink');
            mean_value = mean(data_to_plot, 2);
            
            xx = linspace(100001, max_all_steps, length(data_to_plot));
            xx = xx / 1000; % thousand steps
            
            tx = 1;  
            sem = std(data_to_plot, [], 2) / sqrt(size(data_to_plot, 2)) * tx;  % Standard error of the mean
            
            ciplot(mean_value - sem, mean_value + sem, xx , plot_color)
            hold on
            
            h=plot(xx, mean_value, 'color', plot_color, 'linestyle', line_style);
            
            xlim([100000/1000, max_all_steps/1000])
            
            box off
            grid on
            hold on
    
            if data_type == 1
                hs = [hs, h];
            end
        end
    
        if data_type == 1 && length(hs) == length(explains)
            legend(hs, explains, 'interpreter', 'none');
        end
    
        if data_type == 4
            xlabel('thousand steps')
        end
    
        subplot(N_data_type, 5, [5 * data_type - 1, 5 * data_type - 0])  %right

        p_values = nan(numel(means), numel(means));
    
        [~, p_values(1,2)] = ttest2(stat_data(:, 1, data_type), stat_data(:, 2, data_type), 'Vartype','unequal');
        [~, p_values(2,3)] = ttest2(stat_data(:, 2, data_type), stat_data(:, 3, data_type), 'Vartype','unequal');
        [~, p_values(1,3)] = ttest2(stat_data(:, 1, data_type), stat_data(:, 3, data_type), 'Vartype','unequal');
        
        % Make P symmetric, by copying the upper triangle onto the lower triangle
        lidx = tril(true(size(p_values)), -1);
        PT = p_values';
        p_values(lidx) = PT(lidx);
        
        if data_type == 1
            plineoffset = 12;
        elseif data_type == 2
            plineoffset = 1.5;
        elseif data_type == 3
            plineoffset = 0.15;
        elseif data_type == 4
            plineoffset = 80;
        end
        bar_handle = superbar(means, 'E', errors, 'P', p_values, 'PLineOffset', plineoffset , 'PStarFontSize', 10);
        for iBarSeries = 1:length(ps_str)
            set(bar_handle(iBarSeries), 'FaceColor', plot_colors{iBarSeries}, 'EdgeColor', 'none');
        end
    
        hold on 
        plot(stat_data(:, 1, data_type) - stat_data(:, 1, data_type) + 1, stat_data(:, 1, data_type), 'k.', 'MarkerSize', 6);
        plot(stat_data(:, 2, data_type) - stat_data(:, 2, data_type) + 2, stat_data(:, 2, data_type), 'k.', 'MarkerSize', 6);
        plot(stat_data(:, 3, data_type) - stat_data(:, 3, data_type) + 3, stat_data(:, 3, data_type), 'k.', 'MarkerSize', 6);
        
        if data_type == 1
            ylabel("refractory period (k steps)")
        elseif data_type == 2
            ylabel("maximal prior ln(STD)")
        elseif data_type == 3
            ylabel("minimal habitual ratio")
        elseif data_type == 4
            ylabel("free energy loss")
    
        end
    
        xlim([0.5, 3.5])
        
        xticks([1 2 3])
        xticklabels(explains_short);

        box off

        if data_type == 1
            ylim([0, 58])
        elseif data_type == 2
            ylim([-2, 4])
        elseif data_type == 3
            ylim([0, 0.7])
        elseif data_type == 4
            ylim([3600, 4000])
        end
        drawnow
    
    end
    
    
    % add subtitles
    
    
    annotation('textbox', [0.04, 0.9, 0.1, 0.1], 'String', 'a', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.04, 0.68, 0.1, 0.1], 'String', 'b', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.04, 0.46, 0.1, 0.1], 'String', 'c', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.04, 0.24, 0.1, 0.1], 'String', 'd', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.53, 0.9, 0.1, 0.1], 'String', 'e', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.53 0.68, 0.1, 0.1], 'String', 'f', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.53, 0.46, 0.1, 0.1], 'String', 'g', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    annotation('textbox', [0.53, 0.24, 0.1, 0.1], 'String', 'h', 'LineStyle', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontName', 'Arial', 'FontWeight', 'bold');
    
    
    
    set(gcf, 'color', 'white')

end
