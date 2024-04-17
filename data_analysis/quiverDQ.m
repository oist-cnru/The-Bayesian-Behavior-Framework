function quiverDQ(x, y, relative_arrow_size, min_abs_arrow_size, vcolor, lw)

if ~exist('relative_arrow_size', 'var')
    relative_arrow_size = 0.3;
end

if ~exist('min_abs_arrow_size', 'var')
    absolute_arrow_size = [];
end

if ~exist('vcolor', 'var')
    vc = hsv2rgb([0.6, 0.9, 0.7]); % plot color
else
    vc = vcolor;
end

if ~exist('lw', 'var')
    lw = 1;
end

hold on

x = reshape(x, 1, []);
y = reshape(y, 1, []);

plot(x, y, '-', 'color', vc, 'LineWidth', lw)

% velocity vector
xv = diff(x);
yv = diff(y);

lv = sqrt(xv.^2 + yv.^2); % scalar velocity

arrow_direction = atan2(yv, xv);

if ~isempty(relative_arrow_size) && ~isempty(min_abs_arrow_size)
    vscale = max(relative_arrow_size * lv, min_abs_arrow_size);
elseif ~isempty(relative_arrow_size) && isempty(min_abs_arrow_size)
    vscale = relative_arrow_size * lv;
elseif isempty(relative_arrow_size) && ~isempty(min_abs_arrow_size)
    vscale = min_abs_arrow_size;
end

% vsalce = absolute_arrow_size;

left_arrow_head_x = [x(2:end); x(2:end) - vscale .* cos(arrow_direction + pi/6)];
left_arrow_head_y = [y(2:end); y(2:end) - vscale .* sin(arrow_direction + pi/6)];

right_arrow_head_x = [x(2:end); x(2:end) - vscale .* cos(arrow_direction - pi/6)];
right_arrow_head_y = [y(2:end); y(2:end) - vscale .* sin(arrow_direction - pi/6)];

plot(left_arrow_head_x, left_arrow_head_y, '-', 'color', vc, 'LineWidth', lw)
plot(right_arrow_head_x, right_arrow_head_y, '-', 'color', vc, 'LineWidth', lw)



end