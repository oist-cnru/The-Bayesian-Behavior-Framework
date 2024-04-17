function fig_ratio = plot_diversity_statistics(dataPath)

N = 60; 
n_seeds = 12;

fig_ratio=figure;
set(gcf, 'Position', [400, 200, 400, 400]);

% plot exit ratio

task_name = 'tmaze_habitual';

list = dir(dataPath);

results = zeros(n_seeds, N, 3);
n_left = zeros(n_seeds, 1);
n_right = zeros(n_seeds, 1);
n_fail = zeros(n_seeds, 1);
steps_taken = zeros(n_seeds, N);

for id = 0 : n_seeds -1

    count = 1;

    for e = 0 : N-1

        filename = sprintf('%s_%d_episode_%d.mat', task_name, id, e);
        path = strcat(dataPath, filename);
        if exist(path, 'file')
            data = load(path);
        else
            fprintf("Load failed: %s\n", path)
            continue
        end

        if data.reward(end) >= 80
            if data.info(end, 1) < 0
                n_left(id+1) = n_left(id+1) + 1;
                results(id+1, count, 1:3) = [0.8, 0.3, 0.3]; % left:red 
            else
                n_right(id+1) = n_right(id+1) + 1;
                results(id+1, count, 1:3) = [0.3, 0.3, 0.8]; % right: blue
            end
        else
            n_fail(id + 1) = n_fail(id + 1) + 1;
            results(id+1, count, 1:3) = [0.6, 0.6, 0.6]; % fail, gray

        end
        steps_taken(id + 1, count) = length(data.reward);
        count = count + 1;
        if count > N
            break
        end
    end
    [~, idx] = sort(squeeze(results(id+1, :, 1)), 'descend');
    results(id+1, :, :) = results(id+1, idx, :);
    
end

[~, idx2] = sort(n_left + 0.001 * n_fail);
results(:, :, :) = results(idx2, :, :);

image(results)
ylabel("#agent (sorted)")
xlabel('trial (sorted)')

end