%% Guitarp-Masip et al, 2012

%% Preparation
%close all
clear all

%{

	########     ###    ########     ###    ##     ## ######## ######## ######## ########   ######
	##     ##   ## ##   ##     ##   ## ##   ###   ### ##          ##    ##       ##     ## ##    ##
	##     ##  ##   ##  ##     ##  ##   ##  #### #### ##          ##    ##       ##     ## ##
	########  ##     ## ########  ##     ## ## ### ## ######      ##    ######   ########   ######
	##        ######### ##   ##   ######### ##     ## ##          ##    ##       ##   ##         ##
	##        ##     ## ##    ##  ##     ## ##     ## ##          ##    ##       ##    ##  ##    ##
	##        ##     ## ##     ## ##     ## ##     ## ########    ##    ######## ##     ##  ######

%}

conditions = {'GoToAvoid', 'GoToWin', 'NoGoToAvoid', 'NoGoToWin'};

n_participants = 47;
n_trials = 480;
n_per_cond = n_trials / 4; % how often each condition will be presented

par.p_win = 0.8;
par.p_lose = 0.2;

par.alpha = 0.05; % learning rate

par.delta = 5; % punishment sensitivty
par.gamma = 5; % reward sensitivity

use_zeta = 1; % use action bias?
par.zeta = 0.3; % action bias

use_xi = 1; % use noise?
par.xi = 0.05; % noise

use_pi = 1; % use pavlovian term?
par.pi = 0.5; % avoidance bias/pavlovian

plot_each_part = false; % single plot for each participant
plot_cum_money = false; % example cumulative money earned for one participant


%% Run simulation

%{

	##        #######   #######  ########
	##       ##     ## ##     ## ##     ##
	##       ##     ## ##     ## ##     ##
	##       ##     ## ##     ## ########
	##       ##     ## ##     ## ##
	##       ##     ## ##     ## ##
	########  #######   #######  ##

%}

% get path separator (/ or \), which depends on your operating system
fs = filesep;

% create directories if they do not exist
% warning('off', 'MATLAB:MKDIR:DirectoryExists'); % suppress the warning that the directory exists
mkdir([pwd fs 'plots']);
mkdir([pwd fs 'data']);

% earned_money
money_sep = zeros(n_participants, length(conditions)); % money per trial type separate

% loop over all participants
for i = 1:n_participants

	% initialize empty variables
	list_trialtypes = [ones(n_per_cond, 1); ...
					   ones(n_per_cond, 1) * 2; ...
					   ones(n_per_cond, 1) * 3; ...
					   ones(n_per_cond, 1) * 4]; % choose first, second etc entry of conditions
	list_trialtypes = list_trialtypes(randperm(length(list_trialtypes))); % shuffle trials

	Q = zeros(4, 2); % 4: number of conditions; 2: number of possible choises (go/nogo)
	choice = NaN(n_trials, 1); % what action was chosen in each trial
	action_prob = NaN(n_trials, 2); % choice probabilites (2: go/nogo)
	best_choice = NaN(n_trials, 1); % best choice for each trial
	reward = NaN(n_trials, 1); % reward outcome per trial
	RPE = zeros(n_trials, 2); % reward prediction errors RPE (2:go/nogo)
	V = zeros(n_per_cond, 1); % pavlovian factor
	money = zeros(n_trials, 1); % money the participant earned

	% for the results
	results = cell(n_trials, 8); % 7 entries saved (see end of trial loop)

	% loop over all trials
	for t = 1:n_trials

		% chose what trial type is presented
		curr_trialtype = list_trialtypes(t);

		% define best choice for current trial
		if curr_trialtype == 1
			% go to avoid
			best_choice(t, 1) = 2; % go
		elseif curr_trialtype == 2
			% go to win
			best_choice(t, 1) = 2; % go
		elseif curr_trialtype == 3
			% nogo to avoid
			best_choice(t, 1) = 1; % nogo
		elseif curr_trialtype == 4
			% nogo to win
			best_choice(t, 1) = 1; % nogo
		end

		% action weight for go
		go = exp(Q(curr_trialtype, 2) + use_zeta * par.zeta + use_pi * par.pi * V(curr_trialtype, 1));

		% action weight for nogo
		nogo = exp(Q(curr_trialtype, 1));

		% (action probability softmax) + noise
		action_prob(t, 1) = ((go / (go + nogo)) * (1 - use_xi * par.xi)) + (use_xi * par.xi / 2); % go
		action_prob(t, 2) = ((nogo / (go + nogo)) * (1 - use_xi * par.xi)) + (use_xi * par.xi / 2); % nogo

		% choice made for current trial
		if rand < action_prob(t, 1)
			choice(t, 1) = 2; % go
		else
			choice(t, 1) = 1; % nogo
		end

		% reward
		if choice(t, 1) == best_choice(t, 1)
			% best option chosen
			if curr_trialtype == 2 || curr_trialtype == 4
				% ToWin conditions
				reward(t, 1) = double(rand < par.p_win); % reward 1; 80%
			elseif curr_trialtype == 1 || curr_trialtype == 3
				% ToAvoid condition
				reward(t, 1) = -(double(rand < 1 - par.p_win)); % reward -1; 20%
			end
		else
			% not best option chosen
			if curr_trialtype == 2 || curr_trialtype == 4
				% ToWin conditions
				reward(t, 1) = double(rand < par.p_lose); % reward 1; 20%
			elseif curr_trialtype == 1 || curr_trialtype == 3
				% ToAvoid condition
				reward(t, 1) = -(double(rand < 1 - par.p_lose)); % reward -1; 80%
			end
		end

		% reward prediction errors RPE
		if reward(t, 1) == 1
			% win
			RPE(t, choice(t, 1)) = par.gamma * reward(t, 1) - Q(curr_trialtype, choice(t, 1));
		elseif reward(t, 1) == -1
			% lose
			RPE(t, choice(t, 1)) = par.delta * reward(t, 1) - Q(curr_trialtype, choice(t, 1));
		else
			% 'nothing' option
			RPE(t, choice(t, 1)) = - Q(curr_trialtype, choice(t, 1));
		end

		% pay particpant for current trial
		money(t, 1) = money(t, 1) + reward(t, 1);
		money_sep(i, curr_trialtype) = money_sep(i, curr_trialtype) + reward(t, 1);

		% update Q values
		Q(curr_trialtype, 2) = Q(curr_trialtype, 2) + par.alpha * RPE(t, 2); % go
		Q(curr_trialtype, 1) = Q(curr_trialtype, 1) + par.alpha * RPE(t, 1); % nogo

		% update pavlovian
		V(curr_trialtype, 1) = V(curr_trialtype, 1) + par.alpha * (par.gamma * reward(t, 1) - V(curr_trialtype, 1));

		% save results
		results(t, :) = [t, ... %trial number
						 {conditions{curr_trialtype}}, ... % current trial condition
						 choice(t, 1), ... % selected choice
						 best_choice(t, 1), ... % optimal choice
						 action_prob(t, 1), ... % choice probabilities
						 action_prob(t, 2), ...
						 Q(curr_trialtype, 1), ... % q-value AFTER update
						 money(t, 1)]; % total money earned

	end

	% summary of simulation for current participant
	results_part = table(results(:, 1), ...
						 categorical(results(:, 2), conditions), ...
						 results(:, 3), ...
						 results(:, 4), ...
						 results(:, 5), ...
						 results(:, 6), ...
						 results(:, 7), ...
						 results(:, 8), ...
						 'VariableNames', {'trial', ...
						 				   'condition', ...
						 				   'choice', ...
						 				   'best_choice', ...
						 				   'prob_go', ...
						 				   'prob_nogo', ...
						 				   'Q', ...
						 				   'money'});

	if plot_each_part
		% plot each participant
		fig_part = figure('Name', 'Probability for each condition over all trials');
		for c = 1:4
			curr_cond = conditions{c};
			plot(1:n_per_cond, ...
				 cell2mat(results_part.prob_go(results_part.condition == curr_cond)), ...
				 'DisplayName', curr_cond);
			hold on
		end
		legend
		title('Probability for each condition over all trials')
		xlabel('Trial')
		ylabel('Probability for GO response')

		% save plot
		saveas(fig_part, [pwd fs 'plots' fs 'part_' num2str(i) '.png'])
	end
	% save results
	save([pwd fs 'data' fs 'part_' num2str(i)], 'results_part')

	%close all
end

%% summary of simulation ovver all participants

%{

	 ######     ###    ##     ## ########        ########  ##        #######  ########
	##    ##   ## ##   ##     ## ##         ##   ##     ## ##       ##     ##    ##
	##        ##   ##  ##     ## ##         ##   ##     ## ##       ##     ##    ##
	 ######  ##     ## ##     ## ######   ###### ########  ##       ##     ##    ##
	      ## #########  ##   ##  ##         ##   ##        ##       ##     ##    ##
	##    ## ##     ##   ## ##   ##         ##   ##        ##       ##     ##    ##
	 ######  ##     ##    ###    ########        ##        ########  #######     ##

%}

% read all the data
for part = 1:n_participants
	load([pwd fs 'data' fs 'part_' num2str(part)], 'results_part')
	results_all.GoToAvoid(:, part) = cell2mat(results_part.prob_go(results_part.condition == 'GoToAvoid'));
	results_all.GoToWin(:, part) = cell2mat(results_part.prob_go(results_part.condition == 'GoToWin'));
	results_all.NoGoToWin(:, part) = cell2mat(results_part.prob_go(results_part.condition == 'NoGoToWin'));
	results_all.NoGoToAvoid(:, part) = cell2mat(results_part.prob_go(results_part.condition == 'NoGoToAvoid'));
end

% mean per trial
results_all.GoToAvoid(:, n_participants + 1) = mean(results_all.GoToAvoid(:, 1:n_participants), 2, 'omitnan');
results_all.GoToWin(:, n_participants + 1) = mean(results_all.GoToWin(:, 1:n_participants), 2, 'omitnan');
results_all.NoGoToAvoid(:, n_participants + 1) = mean(results_all.NoGoToAvoid(:, 1:n_participants), 2, 'omitnan');
results_all.NoGoToWin(:, n_participants + 1) = mean(results_all.NoGoToWin(:, 1:n_participants), 2, 'omitnan');

% plot summary of all participants
fig_all = figure('Name', 'Mean probabilities for all participants', ...
				 'Toolbar', 'none');
plot(1:n_per_cond, ...
	 results_all.GoToAvoid(:, n_participants + 1), ...
	 'DisplayName', 'GoToAvoid');
hold on
plot(1:n_per_cond, ...
	 results_all.GoToWin(:, n_participants + 1), ...
	 'DisplayName', 'GoToWin');
plot(1:n_per_cond, ...
	 results_all.NoGoToAvoid(:, n_participants + 1), ...
	 'DisplayName', 'NoGoToAvoid');
plot(1:n_per_cond, ...
	 results_all.NoGoToWin(:, n_participants + 1), ...
	 'DisplayName', 'NoGoToWin');
legend_pos = [.7, .4, .1, .1];
set(legend, 'Position', legend_pos);
title('Mean probabilities of GO response for all participants')
subt = {['n_{Participants} = ' num2str(n_participants) '   and   n_{Trials} = ' num2str(n_trials)], ...
		['p_{win} = ' num2str(par.p_win) '   and   p_{lose} = ' num2str(par.p_lose)], ...
		['Learning rate {\it\alpha} = ' num2str(par.alpha)], ...
		['Reward sensitivity {\it\gamma} = ' num2str(par.gamma) '   and    punishment sensitivity {\it\delta} = ' num2str(par.delta)], ...
		['Action (GO) bias {\it\zeta} = ' num2str(par.zeta * use_zeta) '   and   noise {\it\xi} = ' num2str(par.xi * use_xi) '   and pavlovian bias {\it\pi} = ' num2str(par.pi * use_pi)]};
subtitle(subt);
xlabel('Trial')
ylabel('Probability for GO response')
axis([0 n_per_cond 0 1]);

% save plot
saveas(fig_all, [pwd fs 'plots' fs 'all_participants.png'])

if plot_cum_money
	% plot cumulative money earned
	fig_money = figure('Name', 'Money earned', ...
					   'Toolbar', 'none');
	plot(1:n_trials, ...
		 cumsum(cell2mat(results_part.money)));
	title('Example: Cumulative money earned for one participant')
	xlabel('Trial');
	ylabel('Cumulative money earned')
end
