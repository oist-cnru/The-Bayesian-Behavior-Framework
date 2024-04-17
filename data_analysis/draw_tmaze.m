function draw_tmaze(plot_exit_line)

     if ~exist("plot_exit_line", "var")
         plot_exit_line = 1;
     end
    % ---------------------- Environment Config ----------------------
    
    zoom_coef = 2;
    
    xv = {};
    yv = {};
    cv = {};
    
    % top-left
    xv{1} =  [-5, -4, -4, -5, -5]  * zoom_coef;
    yv{1} =  [-1, -1, 3, 3, -1] * zoom_coef;
    cv{1} = [0.8000, 0.38000, 0.1000];
    
    % top-right
    xv{2} =  [5, 4, 4, 5, 5]  * zoom_coef;
    yv{2} =  [-1, -1, 3, 3, -1] * zoom_coef;
    cv{2} = [0.1000, 0.3800, 0.8000];
    
    % top
    xv{3} =  [2.5, -2.5, -2.5, 2.5, 2.5]  * zoom_coef;
    yv{3} =  [2, 2, 3, 3, 2] * zoom_coef;
    cv{3} = [0.5200, 0.5200, 0.5200];
    
    % middle-left
    xv{4} =  [-1.5, -4, -4, -1.5, -1.5]  * zoom_coef;
    yv{4} =  [-1, -1, 0, 0, -1] * zoom_coef;
    cv{4} = [0.8000, 0.5000, 0.5000];
    
    % middle-right
    xv{5} =  [1.5, 4, 4, 1.5, 1.5]  * zoom_coef;
    yv{5} =  [-1, -1, 0, 0, -1] * zoom_coef;
    cv{5} = [0.5000, 0.5000, 0.8000];
    
    % bottom-left
    xv{6} =  [-1.5, -1.5, -0.5, -0.5, -0.5]  * zoom_coef;
    yv{6} = [-4, 0, 0, -4, -4] * zoom_coef;
    cv{6} = [0.5200, 0.5200, 0.5200];
    
    % bottom-right
    xv{7} =  [1.5, 1.5, 0.5, 0.5, 0.5]  * zoom_coef;
    yv{7} = [-4, 0, 0, -4, -4] * zoom_coef;
    cv{7} = [0.5200, 0.5200, 0.5200];
    
    % bottom
    xv{8} =  [-1.5, -1.5, 1.5, 1.5, 1.5]  * zoom_coef;
    yv{8} = [-4, -5, -5, -4, -4] * zoom_coef;
    cv{8} = [0.7200, 0.2400, 0.7200];

    if plot_exit_line
        % exit line
        xv{9} =  [-4, -2.5, -2.5, -4, -4]  * zoom_coef;
        yv{9} =  [2, 2, 2.05, 2.05, 2] * zoom_coef;
        cv{9} = [0.2200, 0.8200, 0.2200];
        
        % exit line
        xv{10} =  [4, 2.5, 2.5, 4, 4]  * zoom_coef;
        yv{10} =  [2, 2, 2.05, 2.05, 2] * zoom_coef;
        cv{10} = [0.2200, 0.8200, 0.2200];
    end

    hold on
    box off
    axis off
    for i =1 : length(xv)
        fill(xv{i}, yv{i}, cv{i}, 'edgecolor', 'none')
    end

    xlim([-10.5, 10.5])
    ylim([-10.5, 6.5])
end