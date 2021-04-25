function [ax1,ax2,lineLossTotal,lineLossPercept,lineLossNpcc]=initializePlots()

% Initialize training progress plot.
fig1 = figure;

% Double the width and height of the figure.
% fig1.Position(3:4) = 2*fig1.Position(3:4);

ax1 = subplot(2,3,1:3);

% Plot the three losses on the same axes.
hold on
lineLossTotal = animatedline('Color',[0.85 0.325 0.098]);
lineLossPercept = animatedline('Color','r');
lineLossNpcc = animatedline('Color','b');

% Customize appearance of the graph.
legend('Total Loss','Perceptual Loss','NPCC loss','Location','Southwest');
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize image plot.
ax2 = subplot(2,3,4:6);
axis off

end