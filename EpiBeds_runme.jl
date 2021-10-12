# push!(LOAD_PATH,pwd())
# cd("Desktop/Hospitals paper/Julia")
#---
work_dir = "C:/Users/overt/OneDrive/Documents/Work/Manchester_postdoc/Matlab/Hospital/Julia/JuliaMTPcodes/Paper"
cd(work_dir)
println("Start loading packages...")
using DifferentialEquations
using StaticArrays
using StatsPlots; #gr()
using LinearAlgebra
using Distributions
using Random
using Statistics
using DelimitedFiles
using MCMCChains
println("Loading complete!")

# define model and random walk functions
# % State variables: 1-5: S, EA1, EA2, IA, AR; 6-10: ES1, ES2, IS, LR, R;
# % 11-15: LH1, LH2, HR, HC, HD; 16-19: CM, CD, MR, D, N;
# % Rates 1-5: rE, rAR, rLR, rLH, rHR; 6-10: rHC, rHD, rCM, rCD, rMR;
# % Probs: pA, pH, pC, pT, pD
# % Transm: beta, f
#---
function delayed_ode(u,p,t)
  rates, probabilities, beta, f = p
  lambda = beta * ( f*(u[4]+u[5]) + (u[8]+u[9]+u[11]+u[12]) ) / u[20]
  # lambda = beta * ( f*(u[4]+u[5]+u[8]) + (u[9]+u[11]+u[12]) ) / u[20]
  du1  = -u[1] * lambda # S (1)
  du2 = probabilities[1] * u[1] * lambda - rates[1]*u[2] # EA1 (2)
  du3 = rates[1]*u[2] - rates[1]*u[3] # EA2 (3)
  du4 = rates[1]*u[3] - rates[1]*u[4] # IA (4)
  du5 = rates[1]*u[4] - rates[2]*u[5] # AR (5)
  du6  = (1-probabilities[1]) * u[1] * lambda - rates[1]*u[6] # ES1 (6)
  du7  = rates[1]*u[6] - rates[1]*u[7] # ES2 (7)
  du8  = rates[1]*u[7] - rates[1]*u[8] # IS (8)
  du9  = (1-probabilities[2])*rates[1]*u[8] - rates[3]*u[9] # LR (9)
  du10 = rates[2]*u[5] + rates[3]*u[9] + rates[5]*u[13] + rates[10]*u[18] # R (10)
  du11 = probabilities[2]*rates[1]*u[8] - rates[4]*u[11] # LH1 (11)
  du12 = rates[4]*u[11] - rates[4]*u[12] # LH2 (12)
  du13 = (1-probabilities[3]-probabilities[4])*rates[4]*u[12] - rates[5]*u[13] # HR (13)
  du14 = probabilities[3]*rates[4]*u[12] - rates[6]*u[14] # HC (14)
  du15 = probabilities[4]*rates[4]*u[12] - rates[7]*u[15] # HD (15)
  du16 = (1-probabilities[5])*rates[6]*u[14] - rates[8]*u[16] # CM (16)
  du17 = probabilities[5]*rates[6]*u[14] - rates[9]*u[17] # CD (17)
  du18 = rates[8]*u[16] - rates[10]*u[18] # MR (18)
  du19 = rates[7]*u[15] + rates[9]*u[17] # D (19)
  du20 = -rates[7]*u[15] - rates[9]*u[17] # N (20)
  @SVector [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,
   du12,du13,du14,du15,du16,du17,du18,du19,du20]
end

function solve_ode(position,prediction_length::Int64)
  rates = copy(initial_transition_rates);
  transmission_params = copy(initial_transmission_parameters);
  probabilities = copy(initial_probabilities);
  number_of_states = length(initial_state);
  rHR = rates[5];
  rCD = rates[6];
  rCM = rates[7];
  betavec = position[1:length(control_dates)];
  number_of_breakpoints = length(betavec)
  log_initial_infectives = position[number_of_breakpoints+1];
  sigma_hi = position[number_of_breakpoints+2];
  sigma_hp = position[number_of_breakpoints+3];
  sigma_up = position[number_of_breakpoints+4];
  sigma_di = position[number_of_breakpoints+5];
  pC = position[number_of_breakpoints+6];
  pT = position[number_of_breakpoints+7];
  pD = position[number_of_breakpoints+8];
  probabilities[[3,4,5]] = [pC,pT,pD];

  # solve the ODE for the first chunk
  Yt = Array{Float64,2}(undef,number_of_states,length(control_dates)+1)
  Y0 = [initial_population-exp(log_initial_infectives),(1-probabilities[1])*exp(log_initial_infectives),
        0.,0.,0.,probabilities[1]*exp(log_initial_infectives),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,initial_population]
  Yt[:,1] = Y0;
  Yall = Array{Float64,2}(undef,number_of_states,Int(control_dates[end]+1))
  time_range = (0.,control_dates[1])
  t_eval = LinRange(0,control_dates[1],Int(control_dates[1])+1)
  params = [rates,probabilities,transmission_rates]
  prob = ODEProblem(delayed_ode,Y0,time_range,params)
  sol = solve(prob,saveat=t_eval)

  Yall[:,1:Int(control_dates[1])+1] = reduce(hcat, sol.u)
  Yt[:,2] = Yall[:,Int(control_dates[1]+1)]

  for ic in 1:(length(control_dates)-1)
      time_range = (control_dates[ic],control_dates[ic+1]+1)
      t_eval = LinRange(control_dates[ic],
                        control_dates[ic+1]+1,
                        Int(control_dates[ic+1]-control_dates[ic])+2)
      transmission_rates[1] = reduced_beta[ic]
      prob = ODEProblem(delayed_ode,
                        Yt[:,ic+1],
                        time_range,
                        params)
      sol = solve(prob,saveat=t_eval);
      Yall[:,Int(control_dates[ic]):Int(control_dates[ic+1])] = reduce(hcat, sol.u)[:,2:end]
      Yt[:,ic+2] = Yall[:,Int(control_dates[ic+1]-1)]
    end
    eps_val = 1e-6
    Yt[Yt.<eps_val] .= eps_val
    Yall[Yall.<eps_val] .= eps_val
    return Yall
end
#---
function log_likelihood(position::Array{Float64})
if (any(position[collect(Iterators.flatten([1:number_of_breakpoints,
    (number_of_breakpoints+number_data_streams_to_fit+2):(number_of_breakpoints+number_data_streams_to_fit+1+number_probabilities_to_fit)]))] .< 0.0) ||
    any(position[(number_of_breakpoints+2):(number_of_breakpoints+1+number_data_streams_to_fit)] .< 1.0) ||
    any(position[(number_of_breakpoints+number_data_streams_to_fit+2):(number_of_breakpoints+number_data_streams_to_fit+1+number_probabilities_to_fit)] .>= 1.0) ||
    sum(position[(number_of_breakpoints+number_data_streams_to_fit+2):(number_of_breakpoints+number_data_streams_to_fit+3)]) .>= 1.0 )
    # println(position)
    return -Inf, [NaN;NaN;NaN;NaN], NaN
else
    rates = copy(initial_transition_rates);
    transmission_frac = copy(f);
    probabilities = copy(initial_probabilities);
    number_of_states = length(initial_state);
    # % Rates 1-5: rE, rAR, rLR, rLH, rHR; 6-10: rHC, rHD, rCM, rCD, rMR;
    rHR = rates[5];
    rHD = rates[7];
    rCM = rates[8];
    rCD = rates[9];
    betavec = position[1:number_of_breakpoints]
    log_initial_infectives = position[number_of_breakpoints+1];
    sigma_hi = position[number_of_breakpoints+2];
    sigma_hp = position[number_of_breakpoints+3];
    sigma_up = position[number_of_breakpoints+4];
    sigma_di = position[number_of_breakpoints+5];
    # sigma_vec = position[(number_of_breakpoints+2):(number_of_breakpoints+5)]
    pC = position[number_of_breakpoints+6];
    pT = position[number_of_breakpoints+7];
    pD = position[number_of_breakpoints+8];

    #pD = 0.37


    probabilities[[3,4,5]] = [pC,pT,pD];

    # solve the ODE
    Yt = Array{Float64,2}(undef,number_of_states,number_of_breakpoints+1)
    Y0 = [initial_population-exp(log_initial_infectives),(1-probabilities[1])*exp(log_initial_infectives),
          0.,0.,0.,probabilities[1]*exp(log_initial_infectives),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,initial_population]
    Yt[:,1] = Y0;
    Yall = Array{Float64,2}(undef,number_of_states,prediction_length)
    time_range = (1.0,Float64(relative_indices_control_dates[1]))
    t_eval = LinRange(1,relative_indices_control_dates[1],relative_indices_control_dates[1])
    params = [rates,probabilities,betavec[1],transmission_frac]
    prob = ODEProblem(delayed_ode,Y0,time_range,params)
    sol = solve(prob,saveat=t_eval)

    Yall[:,1:relative_indices_control_dates[1]] = reduce(hcat, sol.u)
    Yt[:,2] = Yall[:,relative_indices_control_dates[1]]

    for ic in 1:number_of_breakpoints-1
        time_range = (Float64(relative_indices_control_dates[ic]),Float64(relative_indices_control_dates[ic+1]))
        t_eval = LinRange(relative_indices_control_dates[ic],
                          relative_indices_control_dates[ic+1],
                          relative_indices_control_dates[ic+1]-relative_indices_control_dates[ic]+1)
        params[3] = betavec[ic+1];
        prob = ODEProblem(delayed_ode,
                          Yt[:,ic+1],
                          time_range,
                          params)
        sol = solve(prob,saveat=t_eval);
        Yall[:,relative_indices_control_dates[ic]:relative_indices_control_dates[ic+1]] = reduce(hcat, sol.u)
        Yt[:,ic+2] = Yall[:,relative_indices_control_dates[ic+1]]
    end

    eps_val = 1e-6
    Yt[Yt.<eps_val] .= eps_val
    Yall[Yall.<eps_val] .= eps_val

    Y_hi = reshape(rates[4]*Yall[states_for_hospital_incidence,:],1,prediction_length); # Ensure this is a row vector
    Y_hp = sum(Yall[states_for_hospital_prevalence,:],dims=1); # This is already a row vector as it sums along columns
    Y_up = sum(Yall[states_for_icu_prevalence,:],dims=1);
    Y_di = reshape(rates[7]*Yall[15,:]+rates[9]*Yall[17,:],1,prediction_length);

    # Calculate log likelihood given fitting specification
    # % Rates 1-5: rE, rAR, rLR, rLH, rHR; 6-10: rHC, rHD, rCM, rCD, rMR;
    log_likelihood = 0.0
    # for ds in 1:number_data_streams_to_fit # need to change if we fit only 3 data data_streams_to_fit
    #     for (index,data) in enumerate(all_data_to_fit[ds])
    #         log_likelihood += logpdf(NegativeBinomial(Y_hi[all_indices_to_fit[index]]/(sigma_vec[ds]-1),1/sigma_vec[ds]),data)
    #     end
    if "hospital_incidence" in data_streams_to_fit
      for (index,data) in enumerate(hospital_incidence_data)
        log_likelihood += logpdf(NegativeBinomial(Y_hi[hospital_incidence_indices[index]]/(sigma_hi-1),
                                                  1/sigma_hi),
                                 data)
      end
    end

    if "hospital_prevalence" in data_streams_to_fit
      for (index,data) in enumerate(hospital_prevalence_data)
        log_likelihood += logpdf(NegativeBinomial(Y_hp[hospital_prevalence_indices[index]]/(sigma_hp-1),
                                                  1/sigma_hp),
                                 data)
      end
    end

    if "icu_prevalence" in data_streams_to_fit
      for (index,data) in enumerate(icu_prevalence_data)
        log_likelihood += logpdf(NegativeBinomial(Y_up[icu_prevalence_indices[index]]/(sigma_up-1),
                                                  1/sigma_up),
                                 data)
      end
    end

    if "death_incidence" in data_streams_to_fit
      for (index,data) in enumerate(death_incidence_data)
        log_likelihood += logpdf(NegativeBinomial(Y_di[death_incidence_indices[index]]/(sigma_di-1),
                                                   1/sigma_di),
                                 data)
      end
    end

    # beta(7,7) prior on pD
    #log_likelihood += 6*log(pD) + 6*log(1-pD)
    #log_likelihood += 19*log(pD) + 32*log(1-pD)

    # normally distributed prior (careful as allows negative values)
    if (scenarioname == "Scenario1")
      sigma_prior = 3.35
      mu_prior = 46
    elseif (scenarioname == "Scenario2")
      sigma_prior = 1.8
      mu_prior = 42
    elseif (scenarioname == "Scenario3")
      sigma_prior = 2.15
      mu_prior = 37.2
    elseif (scenarioname == "Scenario4")
      sigma_prior = 1.8
      mu_prior = 35.5
    elseif (scenarioname == "Scenario5")
      sigma_prior = 1.8
      mu_prior = 35.8
    elseif (scenarioname == "Scenario6")
      sigma_prior = 1.7
      mu_prior = 35.7
    elseif (scenarioname == "Scenario7")
      sigma_prior = 3.3
      mu_prior = 15.9
    elseif (scenarioname == "Scenario8")
      sigma_prior = 2.6
      mu_prior = 25.2
    elseif (scenarioname == "Scenario9")
      sigma_prior = 1.95
      mu_prior = 29.3
    elseif (scenarioname == "Scenario10")
      sigma_prior = 1.65
      mu_prior = 30.5
    elseif (scenarioname == "Scenario11")
      sigma_prior = 1.65
      mu_prior = 30.5
    elseif (scenarioname == "Scenario12")
      sigma_prior = 1.65
      mu_prior = 30.5
    end

    log_likelihood += 1*(-1/2*(log(2*pi*sigma_prior^2)) - 1/(2*sigma_prior^2)*(100*pD-mu_prior)^2)

    # Tychonov regularisation step
    # log_likelihood -= 0.1*((sigma_hi-1)^2 + (sigma_hp-1)^2 + (sigma_up-1)^2 + (sigma_di-1)^2)
    return log_likelihood, vcat(Y_hi,Y_hp,Y_up,Y_di), Yall
  end
end
#---
function random_walk(model::Function,
                     number_of_samples::Int64,
                     initial_position::Array{Float64},
                     step_size::Float64,
                     proposal_covariance=I,
                     thinning_rate::Int64=1,
                     store_sims::Bool=false)

    println("Running RWM for $number_of_samples samples")
    # initialise the covariance proposal matrix
    # check if default value is used, and set to q x q identity
    if isequal(proposal_covariance,I)
        identity = true
    else
        identity = false
        proposal_cholesky = cholesky(proposal_covariance).L
    end

    # initialise samples matrix and acceptance ratio counter
    accepted_moves = 0
    mcmc_samples = Array{Float64,2}(undef, number_of_samples, number_of_parameters_to_fit)
    mcmc_samples[1,:] = initial_position
    number_of_iterations = number_of_samples*thinning_rate

    # initial markov chain
    current_position = initial_position
    current_log_likelihood, current_ode_solution, current_full_solution = log_likelihood(current_position)
    # println(current_log_likelihood)
    # println(size(current_ode_solution))

    for iteration_index = 1:number_of_iterations
        if identity
            proposal = current_position + step_size*rand(Normal(),(number_of_parameters_to_fit,1))
        else
            proposal = current_position + step_size*proposal_cholesky*reshape(rand(Normal(),number_of_parameters_to_fit),(number_of_parameters_to_fit,1))
        end
        proposal_log_likelihood, proposed_ode_solution, proposed_full_solution = log_likelihood(proposal)
        if proposal_log_likelihood == -Inf
            if iteration_index%thinning_rate == 0
                mcmc_samples[Int(iteration_index/thinning_rate),:] = current_position
                if store_sims
                  ode_prediction[:,:,Int(iteration_index/thinning_rate)] = current_ode_solution;
                  ode_full_prediction[:,:,Int(iteration_index/thinning_rate)] = current_full_solution;##### HERE
                end
                iteration_index%thinning_rate == 0 && continue
            end
        end

        # accept-reject step
        if rand(Uniform()) < exp(proposal_log_likelihood - current_log_likelihood)
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            current_ode_solution = proposed_ode_solution
            current_full_solution = proposed_full_solution ##### HERE
            accepted_moves += 1
        end
        if iteration_index%thinning_rate == 0
            storing_index = Int(iteration_index/thinning_rate);
            mcmc_samples[storing_index,:] = current_position;
            if store_sims
                ode_prediction[:,:,storing_index] = current_ode_solution;
                ode_full_prediction[:,:,storing_index] = current_full_solution;#### HERE
            end
        end
        if iteration_index%(number_of_iterations/10)==0
            println("Progress: ",100*iteration_index/number_of_iterations,"%   Log-likelihood = ",current_log_likelihood);
            # For monitoring
            # println(current_log_likelihood)
            # println(ode_prediction[:,:,storing_index])
        end
    end # for loop
    println("Acceptance ratio:",accepted_moves/number_of_iterations)
    return mcmc_samples
end # function


# For initial rates and probabilities, k ~ 2.48

#---
#### Define position indicies - Start__________________________________________________
global states_for_hospital_incidence = 12; # LH2 goes into hospital (need to multiply by rate)
global states_for_hospital_prevalence = 13:18;
global states_for_icu_prevalence = 16:17;
global states_for_death_incidence = [15,17]; # Both HD and CD die (need to multiply by rate)
global data_streams_to_fit  = ["hospital_incidence",
            "hospital_prevalence",
            "icu_prevalence",
            "death_incidence"
            ]; # hospital_incidence, hospital_prevalence, icu_prevalence, death_incidence, sero_prevalence
global initial_sigmas = [15.,100.,60.,10.]
#global initial_sigmas = [15.,100.,60.]

global number_data_streams_to_fit = 4;#length(fit); # Note that I still fit 4 sigmas
#### Define position indices - End__________________________________________________

#--- Things to change
######## Options to edit
global initial_run = "NO"
global number_of_samples = 50000;
global number_repeated_fits = 1;
global length_last_fit_window = 21; # Try 4 or 6 weeks
for i = 1:2
  if i == 1
    global mid = 0
  else
    global mid = 15
  end
#global version = "v1"
if (mid == 0)
    global scenarios = 1:10
else
    global scenarios = 1:9
end



# set proportions
pA = 0.55; # Huge range from 0.179 to 0.972 in Davies et al., 2020; Dong et al., 2020; Lavezzo et al., 2020; Mizumoto et al., 2020; Streeck et al., 2020
pH = 0.0312; # Estimated, but in range of Riccardo et al., 2020; Salje et al., 2020; NIPH
pH = 0.036/0.45;
pC = 0.015; # Estimated, but in range of ISARIC; Riccardo et al., 2020; Salje et al., 2020; NIPH
pT = 0.27; # Estimated, but similar to 0.316 of ISARIC
pD = 0.53; # Estimated, but a bit larger than the 0.4-0.45 of ICNARC, 2020; ISARIC
pD = 0.37
global initial_probabilities = [pA,pH,pC,pT,pD];
# set transmission parameters
# (f/rE + pH/rIH + (1-pH)/rIR) + pA*f*(1/rE + 1/rA)
# % State variables: 1-5: S, EA1, EA2, IA, AR; 6-10: ES1, ES2, IS, LR, R;
# % 11-15: LH1, LH2, HR, HC, HD; 16-19: CM, CD, MR, D, N;

#---
for scenario in scenarios
println("Running scenario $scenario")


### What data to discard - Start__________________________________________________
global final_data_to_discard = [2,0,0,5]
if scenario < 7#6 # Include first wave #previously on 6
    global early_data_to_discard = [8,8,8,8]
    #global early_data_to_discard = [18,19,32,0]
    # define initial conditions and parameters
    # set rates
    rEtotal = 1.0/4.84; # Pellis et al. (2020)
    rE = 3.0*rEtotal;
    rAR = 1.0/3.5; # Same as rLR below
    rLR = 1.0/3.5; # Citing a 5-day generation time (including ~1.5 day for IS stage)
    rLHtotal = 1.0/5.2;
    rLH = 2.0*rLHtotal;
    rHR = 1.0/9.59; # NIPH;
    rHC = 1.0/2.88; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/9.05; # estimated
    rCD = 1.0/11.64; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1.0/15.32; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rCM = 1/11.64
    rMR = 1.0/12.1; # estimated, but similar to estiamte from ICNARC, 2020

    rEtotal = 1.0/4.84; # Pellis et al. (2020)
    rE = 3.0*rEtotal;
    rAR = 1.0/3.5; # Same as rLR below
    rLR = 1.0/3.5; # Citing a 5-day generation time (including ~1.5 day for IS stage)
    rLHtotal = 1.0/5.2;
    rLH = 2.0*rLHtotal;
    rHR = 1/10.40; # NIPH;
    rHC = 1/2.36; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/8.75; # estimated
    rCD = 1/10.83; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/16.96; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/9.26; # estimated, but similar to estiamte from ICNARC, 2020

    #global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];
    global f = 0.25; # Fraction reduction in transmission
    global k = (1-pA)*(1/rE + pH/rLHtotal + (1-pH)/rLR) + pA*f*(1/rE + 1/rAR)
    global version = "v2"

else
    #global early_data_to_discard = [8,8,8,8]
    global early_data_to_discard = [8,153,153,153] # Skip first wave in all data streams except hospitalisations
    #global early_data_to_discard = [0,122,122,122] # Skip first wave in all data streams except hospitalisations
    # define initial conditions and parameters
    rEtotal = 1.0/4.84; # Pellis et al. (2020)
    rE = 3.0*rEtotal;
    rAR = 1.0/3.5; # Same as rLR below
    rLR = 1.0/3.5; # Citing a 5-day generation time (including ~1.5 day for IS stage)
    rLHtotal = 1.0/5.2;
    rLH = 2.0*rLHtotal;
    rHR = 1/7.27; # NIPH;
    rHC = 1/2.36; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/10.09; # estimated
    rCD = 1/13.52; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/7.06; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/7.29; # estimated, but similar to estiamte from ICNARC, 2020
    #global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];
    global f = 0.25; # Fraction reduction in transmission
    global k = (1-pA)*(1/rE + pH/rLHtotal + (1-pH)/rLR) + pA*f*(1/rE + 1/rAR)
    global version = "v1"
end

if (version == "v2")
    global data_file_name = "Select_data_130121_paper_v2.csv"
    #global data_file_name = "Select_data_public_chris.csv"
else
    global data_file_name = "Select_data_130121_paper.csv"
    #global data_file_name = "Select_data_public_chris.csv"
end
global regions = ["EN"];#["EE", "LO", "MI", "NE", "NW", "SE", "SW", "UK"];
global scenarioname = string("Scenario",scenario)
global suffix = scenarioname
global susc_frac = zeros(length(regions),20);
global counter = 0;
global commondir = "./All_hospital_outputs/"
if mid == 0
    suffix = string(suffix,"_",version)
    global versiondir = string("./All_hospital_outputs_",version,"/")
else
    suffix = string(suffix,"_mid_",version)
    global versiondir = string("./All_hospital_outputs_mid_",version,"/")
end
global outputdir = string("./",suffix,"/")
if ~isdir(outputdir)
    mkdir(outputdir)
end

if ~isdir(versiondir)
    mkdir(versiondir)
end

# if (scenarioname == "Scenario1")
#     global final_fit_day = 31+mid # End March (or mid April)
#     global control_dates = [24]
#     global names_parameters_to_fit = ["β₁","β₂","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"];
#     global initial_transmission_rates = [1.6,1.]; # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario2")
#     global final_fit_day = 61+mid # End April
#     global control_dates = [24,final_fit_day-length_last_fit_window]
#     global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario3")
#     global final_fit_day = 92+mid # End May
#     global control_dates = [24,final_fit_day-length_last_fit_window]
#     global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario4")
#     global final_fit_day = 122+mid # End June
#     global control_dates = [24,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
#     global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario5")
#     global final_fit_day = 153+mid # End July
#     global control_dates = [24,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
#     global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario6") # start removing 1st wave
#     global final_fit_day = 184+mid # End August
#     global control_dates = [24,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
#     global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario7")
#     global final_fit_day = 214+mid # End Sept
#     global control_dates = [24,163,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
#     global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario8")
#     global final_fit_day = 245+mid # End Oct
#     global control_dates = [24,163,228]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
#     global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# elseif (scenarioname == "Scenario9")
#     global final_fit_day = 275+mid # End Nov
#     global control_dates = [24,163,228,250]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
#     global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
#     global initial_transmission_rates = [1.6,0.3,0.35,0.4,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
# else
#     println("Not sure which scenario you want me to run...")
# end

if (scenarioname == "Scenario1")
    global final_fit_day = 31+mid # End March (or mid April)
    # global final_fit_day = 44-7 # End March (or mid April)
    if mid == 0
      global control_dates = [13,24]
      global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"];
      global initial_transmission_rates = [1.6,1.,1.]; # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
      # global control_dates = [24]
      # global names_parameters_to_fit = ["β₁","β₂","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"];
      # global initial_transmission_rates = [1.6,1.]; # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    else
      global control_dates = [13,24]
      global names_parameters_to_fit = ["β₁","β₂","β₃","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"];
      global initial_transmission_rates = [1.6,1.6,1.]; # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end

    rHR = 1/5.13; # NIPH;
    rHC = 1/1.46; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/8.09; # estimated
    rCD = 1/7.61; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/5.58; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/6.91; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario2")
    global final_fit_day = 61+mid # End April
    # global final_fit_day = 66-14 # End April
    if mid == 0
      global control_dates = [13,24,final_fit_day-length_last_fit_window]
      global names_parameters_to_fit = ["β₁","β₂","β₃","β4","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
      # global control_dates = [24,final_fit_day-length_last_fit_window]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      # global names_parameters_to_fit = ["β₁","β₂","β3","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      # global initial_transmission_rates = [1.6,0.3,0.3] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)

    else
      global control_dates = [13,24,42,final_fit_day-length_last_fit_window]
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β5","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end

    rHR = 1/8.19; # NIPH;
    rHC = 1/2.1; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/7.79; # estimated
    rCD = 1/9.96; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/11.25; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/6.72; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario3")
    global final_fit_day = 92+mid # End May
    global control_dates = [13,24,42,final_fit_day-length_last_fit_window]
    global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β5","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
    global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)

    rHR = 1/8.91; # NIPH;
    rHC = 1/2.49; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/8.45; # estimated
    rCD = 1/11.1; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/16.5; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/9.29; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario4")
    global final_fit_day = 122+mid # End June
    global control_dates = [13,24,42,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
    global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β5","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
    global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)

    rHR = 1/9.2; # NIPH;
    rHC = 1/2.65; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/8.68; # estimated
    rCD = 1/11.4; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/16.63; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/10.76; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario5")
    global final_fit_day = 153+mid # End July
    global control_dates = [13,24,42,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
    global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β5","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
    global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)

    rHR = 1/9.24; # NIPH;
    rHC = 1/2.86; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/8.86; # estimated
    rCD = 1/11.55; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/16.1; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/11.31; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario6") # start removing 1st wave
    global final_fit_day = 184+mid # End August
    if mid == 0
      global control_dates = [13,24,42, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β5","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
      # global initial_transmission_rates = [2.33,1.2,0.26,0.32,0.63] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    else
      global control_dates = [13,24,42,168, final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β5","β6","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end

    rHR = 1/9.35; # NIPH;
    rHC = 1/2.9; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/8.93; # estimated
    rCD = 1/11.84; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/15.93; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/11.86; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario7")
    global final_fit_day = 214+mid # End Sept
    if mid == 0
      global control_dates = [13,24,42,168,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β6","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    else
      global control_dates = [13,24,42,168,190,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β6","β7","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end
    rHR = 1/6.89; # NIPH;
    rHC = 1/1.82; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/5.20; # estimated
    rCD = 1/9.17; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/6.51; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/4.32; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario8")
    global final_fit_day = 245+mid # End Oct
    if mid == 15
      global control_dates = [13,24,42,168,190,228,250]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β₆","β7","β8","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    else
      global control_dates = [13,24,42,168,190,228]#,final_fit_day-length_last_fit_window]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β6","β7","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end

    rHR = 1/10.04; # NIPH;
    rHC = 1/2.62; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/10.46; # estimated
    rCD = 1/14.71; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/6.86; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/6.59; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario9")
    global final_fit_day = 275+mid # End Nov
    if mid == 0
      global control_dates = [13,24,42,168,190,228,250,final_fit_day-length_last_fit_window]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β₆","β7","β8","β9","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4,0.35,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    else ### actual lockdown easing on 277, but gradient change for 272 due to new variant
      global control_dates = [13,24,42,168,190,228,250,263,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β₆","β7","β8","β9","β10","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4,0.35,0.35,0.4] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end
    # global control_dates = [24,50,170,228,250,272]#final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
    # global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β₆","β7","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
    # global initial_transmission_rates = [1.6,0.3,0.35,0.35,0.4,0.4,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)

    rHR = 1/10.07; # NIPH;
    rHC = 1/2.53; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/13.37; # estimated
    rCD = 1/14.88; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/7.46; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/6.27; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario10")
    global final_fit_day = 305+mid # End Dec
    if mid == 0
      global control_dates = [13,24,42,168,190,228,250,263,final_fit_day-length_last_fit_window]#,295]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β₆","β7","β8","β9","β10","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4,0.4,0.4,0.4]#,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    else
      global control_dates = [13,24,42,168,190,228,250,263,311]#,295]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
      global names_parameters_to_fit = ["β₁","β₂","β₃","β₄","β₅","β₆","β7","β8","β9","β10","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
      global initial_transmission_rates = [1.6,1.6,0.3,0.35,0.35,0.4,0.4,0.4,0.4,0.4]#,0.35] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
    end
    rHR = 1/12.16; # NIPH;
    rHC = 1/2.70; # estimated, but compatible with range in ECDC, 2020; ISARIC; Salje et al., 2020; NIPH
    rHD = 1/10.02; # estimated
    rCD = 1/15.33; # estimated, but compatible with range in ECDC, 2020; Grasselli et al., 2020; ICNARC, 2020
    rCM = 1/8.57; # estimated, but compatible with range in ICNARC, 2020; NIPH
    rMR = 1/6.45; # estimated, but similar to estiamte from ICNARC, 2020
    global initial_transition_rates = [rE,rAR,rLR,rLH,rHR,rHC,rHD,rCM,rCD,rMR];

elseif (scenarioname == "Scenario11")
    global final_fit_day = 45 # End Jan
    global control_dates = [24]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
    global names_parameters_to_fit = ["β₁","β₂","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
    global initial_transmission_rates = [1.6,0.3] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
  elseif (scenarioname == "Scenario12")
    global final_fit_day = 66 # End Jan
    global control_dates = [24,final_fit_day-length_last_fit_window]#,final_fit_day-length_last_fit_window] # [24, 50, final_fit_day-length_last_fit_window];
    global names_parameters_to_fit = ["β₁","β₂","β3","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"]
    global initial_transmission_rates = [1.6,0.3,0.3] # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)
else
    println("Not sure which scenario you want me to run...")
    ###[10.,58.,200.,262.,284.,311.,329.,final_day]
end
# global final_fit_day = 60;#168; # NaN; # Change as needed: if NaN, let the imported data tell me
# global control_dates = [24];#[24, 50, final_fit_day-length_last_fit_window];
# global names_parameters_to_fit = ["β₁","β₂","log(J₀)","σ_hi","σ_hp","σ_up","σ_di","pC","pT","pD"];
# global initial_transmission_rates = [1.6,0.3]; # One more than the manually specified control dates (after data is read, the last data of ode is added to coontrol_dates)


#---

#### Adjust time windows - Start__________________________________________________
# To set initial the infection rates, I first need to decide the time windows when they apply
# For convenience, day 0 is set on 29th Feb, and data starts from 1st March (day 1)
# We need to solve the ODEs from earlier, to let numbers of infectives treacle nicely through all compartments, but it's not clear how earlier: let's try 29 days (all of February)
global day_ode_start = -40; # Starting ODEs from end of Jan
global start_data = 1; # Data starts on 1st March
#global length_last_fit_window = 28; # Try 4 or 6 weeks

global length_prediction_window = 30;#42;
global start_nonfit_data = final_fit_day + 1; # If I plot all data, this is NaN
global stop_nonfit_data = final_fit_day + length_prediction_window; # If I plot all data, this is NaN
#### Adjust time windows - End__________________________________________________



# global last_hospital_incidence_data_to_discard = 2;
# global last_death_incidence_data_to_discard = 5;
#### What data to discard - End__________________________________________________


global log_initial_infectives = log(15);
global number_of_parameters_to_fit = length(names_parameters_to_fit);
global number_probabilities_to_fit = 3;

# choose region and fitting criteria
global regions = ["EN"];#["EE", "LO", "MI", "NE", "NW", "SE", "SW", "UK"];

#---
#### Loop and solve over all regions - Start__________________________________________________
for region in regions
    println("Fitting $data_streams_to_fit  to $region")
    global final_fit_day
    global length_last_fit_window
    global control_dates
    global start_data
    global day_ode_start

    #### Define region specific variable - Start__________________________________________________
    if region == "EN"
        # set data
        global data = readdlm(data_file_name,',')[:,29:32]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 56000000.0;


    elseif region == "EE"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,1:4]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 6200000.0;

    elseif region == "LO"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,5:8]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 8900000.0;

    elseif region == "MI"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,9:12]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 10700000.0;

    elseif region == "NE"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,13:16];
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 3210000.0;

    elseif region == "NW"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,17:20]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 7300000.0;

    elseif region == "SE"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,21:24]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 9130000.0;

    elseif region == "SW"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,25:28]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 5600000.0;

    elseif region == "SC"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,33:36]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 5450000.0;

    elseif region == "WA"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,37:40]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 3140000.0;

    elseif region == "NI"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,41:44]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 1880000.0;

    elseif region == "UK"
        # set data
        data = readdlm("all_data_to_export_version2.csv",',')[:,45:48]
        length_data = length(data[:,3]); # Measure the length on ICU prevalence (most reliable, and certainly reaches the end of the data stream)
        if isnan(final_fit_day)
            final_fit_day = length_data;
            control_dates[end] = final_fit_day-length_last_fit_window;
            control_dates = convert(Array{Int,1},control_dates);
        end
        global initial_population = 66470000.0;

    else throw(DomainError("This is not a valid region"))
    end
    #### Define region specific variable - End__________________________________________________

    #### Process data into required formats - Start__________________________________________________
    global hospital_incidence_data  = [data[i,1] for i in 1:final_fit_day if isnan.(data[i,1])==0][(early_data_to_discard[1]+1):(end-final_data_to_discard[1])]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:final_fit_day if isnan.(data[i,2])==0][(early_data_to_discard[2]+1):(end-final_data_to_discard[2])];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:final_fit_day if isnan.(data[i,3])==0][(early_data_to_discard[3]+1):(end-final_data_to_discard[3])];# data[:,3];
    global death_incidence_data     = [data[i,4] for i in 1:final_fit_day if isnan.(data[i,4])==0][(early_data_to_discard[4]+1):(end-final_data_to_discard[4])];# data[:,4];

    global hospital_incidence_indices  = [i+start_data-day_ode_start for i in 1:final_fit_day if isnan.(data[i,1])==0][(early_data_to_discard[1]+1):(end-final_data_to_discard[1])];
    global hospital_prevalence_indices = [i+start_data-day_ode_start for i in 1:final_fit_day if isnan.(data[i,2])==0][(early_data_to_discard[2]+1):(end-final_data_to_discard[2])];
    global icu_prevalence_indices      = [i+start_data-day_ode_start for i in 1:final_fit_day if isnan.(data[i,3])==0][(early_data_to_discard[3]+1):(end-final_data_to_discard[3])];
    global death_incidence_indices     = [i+start_data-day_ode_start for i in 1:final_fit_day if isnan.(data[i,4])==0][(early_data_to_discard[4]+1):(end-final_data_to_discard[4])];


    global all_data_to_fit = Array{Array{Int64,1},1}(undef,number_data_streams_to_fit)
    global all_indices_to_fit = Array{Array{Int64,1},1}(undef,number_data_streams_to_fit)
    global all_data_not_fit = Array{Array{Int64,1},1}(undef,number_data_streams_to_fit)
    global all_indices_not_fit = Array{Array{Int64,1},1}(undef,number_data_streams_to_fit)
    for ds in 1:number_data_streams_to_fit
        data_temp = [data[i,ds] for i in 1:final_fit_day if isnan.(data[i,1])==0]
        all_data_to_fit[ds] = data_temp[(early_data_to_discard[ds]+1):(end-final_data_to_discard[ds])]
        all_data_not_fit[ds] = data_temp[vcat(1:early_data_to_discard[ds],(end-final_data_to_discard[ds]+1):end)]
        indices_temp = [i+start_data-day_ode_start for i in 1:final_fit_day if isnan.(data[i,1])==0]
        all_indices_to_fit[ds] = indices_temp[(early_data_to_discard[ds]+1):(end-final_data_to_discard[ds])]
        all_indices_not_fit[ds] = indices_temp[vcat(1:early_data_to_discard[ds],(end-final_data_to_discard[ds]+1):end)]
    end
    ##### Process data into required formats - End__________________________________________________


    #### Set initial conditions - Start__________________________________________________
    global indices = Vector((start_data-day_ode_start+1):(start_data-day_ode_start+final_fit_day))
    global final_prediction_day = final_fit_day+length_prediction_window;
    global control_dates = collect(Iterators.flatten([control_dates,final_prediction_day]));
    global number_of_breakpoints = length(control_dates);
    global relative_indices_control_dates = control_dates.-day_ode_start.+1; # Vectors start from 1, so relative control dates (with ODEs starting at 0) would be this - 1
    global prediction_length = final_prediction_day-day_ode_start+1; # This is also the last relative index of control date
    global initial_state = @SVector [initial_population-exp(log_initial_infectives),(1-pA)*exp(log_initial_infectives),
          0.,0.,0.,pA*exp(log_initial_infectives),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,initial_population];
    global parameters = [initial_transition_rates,initial_probabilities,initial_transmission_rates];
    global initial_parameters_to_fit = reshape(collect(Iterators.flatten([initial_transmission_rates,[log_initial_infectives],initial_sigmas,initial_probabilities[3:5]])),:,1);

    ### testing testing
    global initial_parameters_to_fit2 = initial_parameters_to_fit #+ 0.5*rand(Normal(),(length(initial_parameters_to_fit),1))

    global data_to_print = fill(NaN,prediction_length,number_data_streams_to_fit)


    # println(size(indices))
    # println(size(data))
    # println(size(data_to_print))
    data_to_print[indices,:] .= data[1:length(indices),:]
    #### Set initial conditions - End__________________________________________________


    #### Run the MCMC - Start__________________________________________________
    number_of_states = length(initial_state);
    global ode_prediction = Array{Float64,3}(undef,number_data_streams_to_fit,prediction_length,number_of_samples); # 4 data streams
    global ode_full_prediction = Array{Float64,3}(undef,number_of_states,prediction_length,number_of_samples); # 4 data streams

    if initial_run == "YES"
        # do this for an initial run to get a variance covariance matrix
        output = random_walk(delayed_ode,
                             number_of_samples,
                             initial_parameters_to_fit, # [0.67,0.2,0.2,0.2,2.0,10.0,10.0,10.0,10.0,0.5,0.5,0.5],
                             0.001)
        covariance_matrix = cov(output[Int(number_of_samples/2):end,:]);
        initial_value = reshape(mean(output[Int(number_of_samples/2):end,:],dims=1),:,1);
        output = random_walk(delayed_ode,
                             number_of_samples,
                             initial_value,
                             0.5,
                             covariance_matrix,
                             1)
        covariance_matrix = cov(output[Int(number_of_samples/2):end,:]);
        initial_value = vec(mean(output[Int(number_of_samples/2):end,:],dims=1));
        posterior_samples = random_walk(delayed_ode,
                                        number_of_samples,
                                        initial_value,
                                        0.55,
                                        covariance_matrix,
                                        1,
                                        true) # Store simulation output
        println("Saving output file...")
        writedlm(string("$region","_output_probs_",suffix,".csv"),posterior_samples,',')

        ### testing testing
        output2 = random_walk(delayed_ode,
                             number_of_samples,
                             initial_parameters_to_fit2, # [0.67,0.2,0.2,0.2,2.0,10.0,10.0,10.0,10.0,0.5,0.5,0.5],
                             0.001)
        covariance_matrix2 = cov(output2[Int(number_of_samples/2):end,:]);
        initial_value2 = reshape(mean(output2[Int(number_of_samples/2):end,:],dims=1),:,1);
        output2 = random_walk(delayed_ode,
                             number_of_samples,
                             initial_value2,
                             0.5,
                             covariance_matrix2,
                             1)
        covariance_matrix2 = cov(output2[Int(number_of_samples/2):end,:]);
        initial_value2 = vec(mean(output2[Int(number_of_samples/2):end,:],dims=1));
        posterior_samples2 = random_walk(delayed_ode,
                                        number_of_samples,
                                        initial_value2,
                                        0.55,
                                        covariance_matrix2,
                                        1,
                                        true) # Store simulation output
        println("Saving output file...")
        writedlm(string("$region","_output_probs2_",suffix,".csv"),posterior_samples2,',')
    elseif initial_run == "NO"
        # do this once you have a covariance matrix for decent posteriors
        for count in 1:number_repeated_fits
            output = readdlm(string("$region","_output_probs_",suffix,".csv"),',');
            covariance_matrix = cov(output[Int(size(output)[1]/2):end,:]);
            initial_value = vec(mean(output[Int(size(output)[1]/2):end,:],dims=1));
            posterior_samples = random_walk(delayed_ode,
                                          number_of_samples,
                                          initial_value,
                                          0.55,
                                          covariance_matrix,
                                          1,
                                          true)
            #testing testing
            output2 = readdlm(string("$region","_output_probs2_",suffix,".csv"),',');
            covariance_matrix2 = cov(output2[Int(size(output2)[1]/2):end,:]);
            initial_value2 = vec(mean(output2[Int(size(output2)[1]/2):end,:],dims=1));
            posterior_samples2 = random_walk(delayed_ode,
                                          number_of_samples,
                                          initial_value2,
                                          0.55,
                                          covariance_matrix2,
                                          1,
                                          true)

            println("Saving output file...")
            writedlm(string("$region","_output_probs_",suffix,".csv"),posterior_samples,',')
            writedlm(string("$region","_output_probs2_",suffix,".csv"),posterior_samples2,',')

            # chain = Chains(posterior_samples,names_parameters_to_fit);

            chain = Chains(cat(posterior_samples,posterior_samples2;dims = 3),names_parameters_to_fit);

            println("Plotting chains...")
            savefig(plot(chain),string(outputdir,"$region","_chain_probs_",suffix,".png"))
            savefig(plot(chain),string(versiondir,"$region","_chain_probs_",suffix,".png"))
            savefig(plot(chain),string(commondir,"$region","_chain_probs_",suffix,".png"))

            # savefig(corner(chain),string(outputdir,"$region","_corner_",suffix,".png"))
            # savefig(corner(chain),string(versiondir,"$region","_corner_",suffix,".png"))
            # savefig(corner(chain),string(commondir,"$region","_corner_",suffix,".png"))
            println("Done!")

        end
    else throw(DomainError("Is this the initial run?"))
    end
    #### Run the MCMC - End__________________________________________________

    #### Save output traces - Start__________________________________________________
    global posterior_samples = readdlm(string("$region","_output_probs_",suffix,".csv"),',');
    global posterior_samples2 = readdlm(string("$region","_output_probs2_",suffix,".csv"),',');


    println("Done!")

    chain = Chains(posterior_samples,names_parameters_to_fit);

    ## testign testing
    chain = Chains(cat(posterior_samples,posterior_samples2;dims = 3),names_parameters_to_fit);

    println("Plotting chains...")
    savefig(plot(chain),string(outputdir,"$region","_chain_probs_",suffix,".png"))
    savefig(plot(chain),string(versiondir,"$region","_chain_probs_",suffix,".png"))
    savefig(plot(chain),string(commondir,"$region","_chain_probs_",suffix,".png"))
    println("Done!")

    #### Save output traces - End__________________________________________________
    ### testing testing
    # posterior_samples = cat(posterior_samples,posterior_samples2;dims = 1)

    #### Generate negative binomial noise - Start__________________________________________________
    println("Simulating prediction noise...")
    # draw negative binomial samples using ode_prediction
    simulated_prediction = Array{Float64,3}(undef,number_data_streams_to_fit,prediction_length,number_of_samples)
    for ds in 1:number_data_streams_to_fit
        simulated_prediction[ds,1,:] = zeros(1,number_of_samples)
        for day in 2:prediction_length
            r = ode_prediction[ds,day,:]./(posterior_samples[:,number_of_breakpoints+1+ds].-1)
            # println(r)
            p = 1 ./ posterior_samples[:,number_of_breakpoints+1+ds]
            # println(p)
            for index in 1:number_of_samples
                if (r[index] == 0)
                    println(string("Problem at ",[ds,day,index]))
                    fff;
                end
                simulated_prediction[ds,day,index] = rand(NegativeBinomial(r[index],p[index]))
            end
        end
    end
    println("Done!")
    #### Generate negative binomial noise - End__________________________________________________

    #### Save output files - Start__________________________________________________
    println("Writing predictions...")
    all_quantiles = [0.05,0.25,0.5,0.75,0.95]
    number_all_quantiles = length(all_quantiles)
    all_quantiles_ode = Array{Float64,3}(undef,number_data_streams_to_fit,prediction_length,number_all_quantiles);
    all_quantiles_sim = Array{Float64,3}(undef,number_data_streams_to_fit,prediction_length,number_all_quantiles);
    # Structure of all_predictions_to_print: first column is just counter (0 at data start); then, for each data stream: data all/not_fitted/fitted (3 column), quantiles_sim (usually 5 columns), quantiles_ode (usually 5 columns)
    blocksize = (2*number_all_quantiles+3)
    global all_predictions_to_print = Array{Float64,2}(undef,prediction_length,1+blocksize*number_data_streams_to_fit)
    all_predictions_to_print[:,1] = day_ode_start:final_prediction_day;
    short_quantiles = [0.05,0.5,0.95]
    number_short_quantiles = length(short_quantiles)
    # short_quantiles_ode = Array{Float64,3}(undef,number_data_streams_to_fit,prediction_length,number_short_quantiles);
    short_quantiles_sim = Array{Float64,3}(undef,number_data_streams_to_fit,prediction_length,number_short_quantiles);
    # Structure of short_predictions_to_print: first column is just counter (0 at data start); then, for each data stream: data fitted (1 column), short_quantiles_sim (usually 3 columns)
    short_blocksize = 1+number_short_quantiles
    global short_predictions_to_print = Array{Float64,2}(undef,prediction_length,1+short_blocksize*number_data_streams_to_fit)
    short_predictions_to_print[:,1] = day_ode_start:final_prediction_day;
    # Parameter posterior quantiles
    for ds in 1:number_data_streams_to_fit
        # One column with the data (first block starts from from column 2)
        all_predictions_to_print[:,1+(ds-1)*blocksize+1] = data_to_print[:,ds]
        all_predictions_to_print[:,1+(ds-1)*blocksize+2] = fill(NaN,prediction_length)
        all_predictions_to_print[:,1+(ds-1)*blocksize+3] = fill(NaN,prediction_length)
        all_predictions_to_print[all_indices_not_fit[ds],1+(ds-1)*blocksize+2] = data_to_print[all_indices_not_fit[ds],ds]
        all_predictions_to_print[all_indices_to_fit[ds],1+(ds-1)*blocksize+3] = data_to_print[all_indices_to_fit[ds],ds]
        for iaq in 1:number_all_quantiles
            all_quantiles_ode[ds,:,iaq] = [quantile(ode_prediction[ds,i,:],all_quantiles[iaq]) for i in 1:prediction_length]
            all_quantiles_sim[ds,:,iaq] = [quantile(simulated_prediction[ds,i,:],all_quantiles[iaq]) for i in 1:prediction_length]
            all_predictions_to_print[:,1+(ds-1)*blocksize+3+iaq] = transpose(all_quantiles_sim[ds,:,iaq])
            # println(size(all_predictions_to_print))
            # println(1+(ds-1)*blocksize+3+number_all_quantiles+iaq)
            all_predictions_to_print[:,1+(ds-1)*blocksize+3+number_all_quantiles+iaq] = transpose(all_quantiles_ode[ds,:,iaq])
        end
        short_predictions_to_print[:,1+(ds-1)*short_blocksize+1] = fill(NaN,prediction_length)
        short_predictions_to_print[all_indices_to_fit[ds],1+(ds-1)*short_blocksize+1] = data_to_print[all_indices_to_fit[ds],ds]
        for isq in 1:number_short_quantiles
            # short_quantiles_ode[ds,:,isq] = [quantile(ode_prediction[ds,i,:],short_quantiles[isq]) for i in 1:prediction_length]
            short_quantiles_sim[ds,:,isq] = [quantile(simulated_prediction[ds,i,:],short_quantiles[isq]) for i in 1:prediction_length]
            short_predictions_to_print[:,1+(ds-1)*short_blocksize+1+isq] = transpose(short_quantiles_sim[ds,:,isq])
        end
    end
    writedlm(string(outputdir,"$region","_predictions_all_",suffix,".csv"),all_predictions_to_print,',')
    writedlm(string(outputdir,"$region","_predictions_short_",suffix,".csv"),short_predictions_to_print,',')
    writedlm(string(versiondir,"$region","_predictions_all_",suffix,".csv"),all_predictions_to_print,',')
    writedlm(string(versiondir,"$region","_predictions_short_",suffix,".csv"),short_predictions_to_print,',')
    writedlm(string(commondir,"$region","_predictions_all_",suffix,".csv"),all_predictions_to_print,',')
    writedlm(string(commondir,"$region","_predictions_short_",suffix,".csv"),short_predictions_to_print,',')

    parameter_names_to_print = names_parameters_to_fit
    parameter_names_to_print[1:number_of_breakpoints] = [string("R",i) for i in 1:number_of_breakpoints]
    parameter_names_to_print[number_of_breakpoints+1] = "Log(J0)"
    parameter_names_to_print[(number_of_breakpoints+2):(number_of_breakpoints+1+number_data_streams_to_fit)] = ["sigma_hi","sigma_hp","sigma_up","sigma_di"]
    global parameters_quantiles = Array{Any,2}(undef,1+number_of_parameters_to_fit,1+number_all_quantiles)
    parameters_quantiles[1,1] = "Parameter"
    parameters_quantiles[1,2:end] = [string("Q ",all_quantiles[i]) for i in 1:number_all_quantiles]
    global parameters_CIs = Array{Float64,2}(undef,number_of_parameters_to_fit,number_short_quantiles)
    for ip in 1:number_of_parameters_to_fit
        parameters_quantiles[ip+1,1] = parameter_names_to_print[ip]
        if (ip <= number_of_breakpoints)
            ######## Add suscpectible depletion here_________________________________________________________________________________________________________________________________
            parameters_quantiles[ip+1,2:end] = [quantile(posterior_samples[:,ip],all_quantiles[i])*k for i in 1:number_all_quantiles]
        else
            parameters_quantiles[ip+1,2:end] = [quantile(posterior_samples[:,ip],all_quantiles[i]) for i in 1:number_all_quantiles]
        end
        parameters_CIs[ip,1] = quantile(posterior_samples[:,ip],0.5)
        parameters_CIs[ip,2] = quantile(posterior_samples[:,ip],short_quantiles[1])
        parameters_CIs[ip,3] = quantile(posterior_samples[:,ip],short_quantiles[3])
    end
    writedlm(string(outputdir,"$region","_parameters_all_",suffix,".csv"),parameters_quantiles,',')
    writedlm(string(versiondir,"$region","_parameters_all_",suffix,".csv"),parameters_quantiles,',')
    writedlm(string(commondir,"$region","_parameters_all_",suffix,".csv"),parameters_quantiles,',')
    if (short_quantiles[1] == 0.025)
        writedlm(string(outputdir,"$region","_parameters_95CIs_",suffix,".csv"),parameters_CIs,',')
        writedlm(string(versiondir,"$region","_parameters_95CIs_",suffix,".csv"),parameters_CIs,',')
        writedlm(string(commondir,"$region","_parameters_95CIs_",suffix,".csv"),parameters_CIs,',')
    elseif (short_quantiles[1] == 0.05)
        writedlm(string(outputdir,"$region","_parameters_90CIs_",suffix,".csv"),parameters_CIs,',')
        writedlm(string(versiondir,"$region","_parameters_90CIs_",suffix,".csv"),parameters_CIs,',')
        writedlm(string(commondir,"$region","_parameters_90CIs_",suffix,".csv"),parameters_CIs,',')
    else
        println("Not sure which credible intervals you want, but I'm spitting them out anyway...")
        writedlm(string(outputdir,"$region","_parameters_CIs_",suffix,".csv"),parameters_CIs,',')
        writedlm(string(versiondir,"$region","_parameters_CIs_",suffix,".csv"),parameters_CIs,',')
        writedlm(string(commondir,"$region","_parameters_CIs_",suffix,".csv"),parameters_CIs,',')
    end

    println("Done!")

    println("Plotting...")
    for ds in 1:number_data_streams_to_fit
        plot(day_ode_start:final_prediction_day,short_quantiles_sim[ds,:,2],color="#20948B",alpha=0.4)
        plot!(day_ode_start:final_prediction_day,short_quantiles_sim[ds,:,3],color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
        plot!(day_ode_start:final_prediction_day,short_quantiles_sim[ds,:,1],color="#20948B",linestyle=:dash,alpha=0.4)
        plot!(all_indices_to_fit[ds].+(day_ode_start-1),all_data_to_fit[ds],seriestype=:scatter,color="black",markersize=3)
        plot!(all_indices_not_fit[ds].+(day_ode_start-1),all_data_not_fit[ds],seriestype=:scatter,markercolor="white",markerstrokecolor="black",markersize=3) #markershape=:diamond
        savefig(string(outputdir,"$region","_",data_streams_to_fit[ds],"_",suffix,".pdf"))
        savefig(string(versiondir,"$region","_",data_streams_to_fit[ds],"_",suffix,".pdf"))
        savefig(string(commondir,"$region","_",data_streams_to_fit[ds],"_",suffix,".pdf"))
        # if (ds == 1)
        #     plot!(hospital_incidence_indices.+(day_ode_start-1),hospital_incidence_data, seriestype=:scatter, color="black",markersize=3)
        #     savefig(string("$region","_posterior_hospital_incidence_probs_version2.pdf"))
        # elseif (ds == 2)
        #     plot!(hospital_prevalence_indices.+(day_ode_start-1),hospital_prevalence_data, seriestype=:scatter, color="black",markersize=3)
        #     savefig(string("$region","_posterior_hospital_prevalence_probs_version2.pdf"))
        # elseif (ds == 3)
        #     plot!(icu_prevalence_indices.+(day_ode_start-1),icu_prevalence_data, seriestype=:scatter, color="black",markersize=3)
        #     savefig(string("$region","_posterior_icu_prevalence_probs_version2.pdf"))
        # else
        #     plot!(death_incidence_indices.+(day_ode_start-1),death_incidence_data, seriestype=:scatter, color="black",markersize=3)
        #     savefig(string("$region","_posterior_death_probs_version2.pdf"))
        # end
    end
    println("Done!")

    # compute the fractioon still susceptible
    number_of_days = prediction_length+1;
    ode_full_prediction = permutedims(ode_full_prediction,[3,1,2])
    global counter += 1;
    all_susc_frac = (ode_full_prediction[:,1,final_fit_day] ./ initial_population)
    susc_frac_values = transpose(quantile(all_susc_frac,0.05:0.05:0.95))
    susc_frac[counter,1] = susc_frac_values[10]
    susc_frac[counter,2:20] = susc_frac_values

    # R history
    susc_frac_over_time = zeros(number_of_days,20)
    Rhistory = zeros(final_fit_day,20)
    for i in 10:(number_of_days-5)
      temp = transpose(quantile((ode_full_prediction[:,1,i] ./ initial_population),0.05:0.05:0.95))
      susc_frac_over_time[i,1] = temp[10]
      susc_frac_over_time[i,2:20] = temp
    end
    for i in 1:final_fit_day
      # for j in 1:length(initial_control_dates)
      #   if initial_control_dates[j]
      count = findfirst(control_dates .> i-1)
      # println([i,count-1,mean(posterior_samples[:,count-1])])
        # if ( i < initial_control_dates[count] )
        #   println(posterior_samples[:,count-1]*b*k)
          temp = transpose(quantile(( (posterior_samples[:,count]*k) .* ode_full_prediction[:,1,i] ./ initial_population ),0.05:0.05:0.95))
          Rhistory[i,1] = temp[10]
          Rhistory[i,2:20] = temp
        # else
        #   count =
        # end
      # end
    end
    writedlm(string(outputdir,"Susc_","$region",suffix,".csv"),susc_frac_over_time,',')
    writedlm(string(outputdir,"Rhistory_","$region",suffix,".csv"),Rhistory,',')

    #### Save output files - End__________________________________________________

    # # hospital incidence
    # lower  = [quantile(prediction[1,i,:],0.025) for i in 1:prediction_length];
    # upper = [quantile(prediction[1,i,:],0.975) for i in 1:prediction_length];
    # median = [quantile(prediction[1,i,:],0.5) for i in 1:prediction_length];
    # plot(day_ode_start:final_prediction_day,upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
    # plot!(day_ode_start:final_prediction_day,lower,color="#20948B",linestyle=:dash,alpha=0.4)
    # plot!(day_ode_start:final_prediction_day,median,color="#20948B",alpha=0.4)
    # plot!(hospital_incidence_indices.+(day_ode_start-1),hospital_incidence_data, seriestype=:scatter, color="black",markersize=3)
    # savefig(string("$region","_posterior_hospital_incidence_probs_version2.pdf"))
    #
    # # hospital prevalence
    # lower = [quantile(prediction[2,i,:],0.025) for i in 1:prediction_length];
    # upper = [quantile(prediction[2,i,:],0.975) for i in 1:prediction_length];
    # median = [quantile(prediction[2,i,:],0.5) for i in 1:prediction_length];
    # plot(day_ode_start:final_prediction_day,upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
    # plot!(day_ode_start:final_prediction_day,lower,color="#20948B",linestyle=:dash,alpha=0.4)
    # plot!(day_ode_start:final_prediction_day,median,color="#20948B",alpha=0.4)
    # plot!(hospital_prevalence_indices.+(day_ode_start-1),hospital_prevalence_data, seriestype=:scatter, color="black",markersize=3)
    # savefig(string("$region","_posterior_hospital_prevalence_probs_version2.pdf"))
    #
    # # icu prevalence
    # lower = [quantile(prediction[3,i,:],0.025) for i in 1:prediction_length];
    # upper = [quantile(prediction[3,i,:],0.975) for i in 1:prediction_length];
    # median = [quantile(prediction[3,i,:],0.5) for i in 1:prediction_length];
    # plot(day_ode_start:final_prediction_day,upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
    # plot!(day_ode_start:final_prediction_day,lower,color="#20948B",linestyle=:dash,alpha=0.4)
    # plot!(day_ode_start:final_prediction_day,median,color="#20948B",alpha=0.4)
    # plot!(icu_prevalence_indices.+(day_ode_start-1),icu_prevalence_data, seriestype=:scatter, color="black",markersize=3)
    # savefig(string("$region","_posterior_icu_prevalence_probs_version2.pdf"))
    #
    # # death incidence
    # lower = [quantile(prediction[4,i,:],0.025) for i in 1:prediction_length];
    # upper = [quantile(prediction[4,i,:],0.975) for i in 1:prediction_length];
    # median = [quantile(prediction[4,i,:],0.5) for i in 1:prediction_length];
    # plot(day_ode_start:final_prediction_day,upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
    # plot!(day_ode_start:final_prediction_day,lower,color="#20948B",linestyle=:dash,alpha=0.4)
    # plot!(day_ode_start:final_prediction_day,median,color="#20948B",alpha=0.4)
    # plot!(death_incidence_indices.+(day_ode_start-1),death_incidence_data, seriestype=:scatter, color="black",markersize=3)
    # savefig(string("$region","_posterior_death_probs_version2.pdf"))
    # println("Done!")
    #
    # println(lower)
    # println(day_ode_start:final_prediction_day)
    # println(death_incidence_indices.+(day_ode_start-1))
    # println(death_incidence_data)


end

# writedlm(string("Randr_value_adjusted_version2.csv"),allRandgrowth,',');

end


end #### Loop over mid scenario
