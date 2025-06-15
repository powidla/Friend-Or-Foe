%script takes a few particular metabolisms and runs through all pairings in
%some random environments

%algorithm:
%select ALWAYS essential compounds
%randomly sample nEnvs different environments size nECinEnv
nEnvSamples = 500;
nECinEnv = 250;
conc = -1000;
%for input metabolisms, calculate growth rates (in isolation and when 
%together maximising A or B), interactions, EC fluxes and rxn fluxes for
%all possible pairings in that database

%% set up:
pathWcommonfuns = '/Users/sasha/Documents/FriendVsFoeProject/code/common_functions/';
addpath(pathWcommonfuns)

% filedr_FvFProjData = '/Users/josephinesolowiej-wedderburn/Documents/FriendVsFoeProject/data/EnvsForSpecialPairs_fromBigRun100/';

% %%collection: AGORA
collection = 'AGORA';     %which collection are we in
colAbr = 'AG';
nMs = 818;
nECtot = 424;
% % % % % nMs = 10;
M_spec = [569, 505, 244, 355, 678];
% % M_spec = [569, 678];

%%collection: CarveMe
% collection = 'CarveMe';     %which collection are we in
% colAbr = 'CM';
% nMs = 5587;
% nECtot = 499;
% M_spec = [3404, 2057, 4618, 3789, 2489];

filedr_metmodels =  ['/Users/sasha/Documents/General_MetModels/Data/' collection '_HGTmodels/'];

% savestr_start = ['/Users/sasha/Documents/FriendVsFoeProject/data/TestRunFeb2025_SpecIDs_withAll_someEnvs/', colAbr];
savestr_start = ['/Users/sasha/Documents/FriendVsFoeProject/data/Run_SpecIDs_withAll_500Envs/', colAbr];

%% which compounds are always essential for all compounds in collection
filedr_gen =  '/Users/sasha/Documents/General_MetModels/Data/';
load([filedr_gen, 'metab_models_extras/', colAbr, 'EssECs.mat'])

%identify always essential compounds:
EC_vec = 1:nECtot;
alwaysEssential = EC_vec(sum(EC_Mat,2)==nMs);
otherECs = EC_vec;
otherECs(alwaysEssential) = [];

%% generate random environments

env_mat = sparse(nEnvSamples, nECtot);
env_mat(:,alwaysEssential) = 1;

for ii = 1:nEnvSamples
    randSample = datasample(otherECs, nECinEnv-length(alwaysEssential), Replace=false);
    env_mat(ii,randSample)=1;
end

%%save environments:
savestr = [savestr_start, 'env_mat.mat'];
save(savestr, 'env_mat' ) 

%% info on met models this collection:

Mi = getHGTmodel_fromFile( 1, filedr_metmodels );
num_nec = length(Mi.rhs_int_lb);  %no. of compounds not in the environment
nRxns = length(Mi.lb);

%% for each environment, need to know the individual growth rate of each microbe:
%%%NOTE%%%
%%%set the env_rhs_ub = individual bound for each individual growth rate,
%%%and combined sum when grown together;
%%%this is different to what was done for env study where the rhs_ub was
%%%always the combined total for a given pair;
%%%but should speed up computation when looking at different individuals in
%%%different envs

indiv_GRs = zeros(nEnvSamples,nMs);
indiv_ECfluxes = repmat({sparse(nEnvSamples,nECtot)},1,nMs);
indiv_rxnfluxes = repmat({sparse(nEnvSamples,nRxns)},1,nMs);

tic
for mm = 1:nMs

    if mod(mm, 100)==0
        disp(mm)
    end

    ind = mm;
    Mi = getHGTmodel_fromFile( ind, filedr_metmodels );

    env_rhsub = Mi.rhs_ext_ub;
    tempmodelMi = createTempmetmodelMi_noenvrhslbs( Mi, env_rhsub, nECtot, num_nec );

    temp_ECfluxes = sparse(nEnvSamples,nECtot);
    nRxns_Mi = length(Mi.lb);
    temp_rxnfluxes = sparse(nEnvSamples,nRxns_Mi);

    for ee = 1:nEnvSamples
        env_rhslb = conc*env_mat(ee,:)';
        output_JustMi = runTempmodelMi_extras( tempmodelMi, nECtot, env_rhslb );

        indiv_GRs(ee,mm) = output_JustMi.RateMi;
        temp_ECfluxes(ee,:) = output_JustMi.ECfluxMat';
        temp_rxnfluxes(ee,:) = output_JustMi.rxnfluxes';
    end

    indiv_ECfluxes{mm} = temp_ECfluxes;
    indiv_rxnfluxes{mm} = temp_rxnfluxes;

end
toc

%%save indiv microbe data:
savestr = [savestr_start, 'indiv_GRs.mat'];
save(savestr, 'indiv_GRs' ) 
savestr = [savestr_start, 'indiv_ECfluxes.mat'];
save(savestr, 'indiv_ECfluxes' ) 
savestr = [savestr_start, 'indiv_rxnfluxes.mat'];
save(savestr, 'indiv_rxnfluxes' ) 

%% for each of the specified microbes, run through each possible pairing and try them in each of the random environments

%set up to save:
%growth rates:
GR_cell = repmat({sparse(nEnvSamples,6)},length(M_spec),nMs);
%EC fluxes:
ECfluxes_maxM1wM2noworse_cell = repmat({sparse(nEnvSamples,nECtot)},length(M_spec),nMs);
ECfluxes_maxM2wM1noworse_cell = repmat({sparse(nEnvSamples,nECtot)},length(M_spec),nMs);
M1rxnFluxes_maxM1wM2noworse_cell = cell(length(M_spec),nMs);
M2rxnFluxes_maxM1wM2noworse_cell = cell(length(M_spec),nMs);
M1rxnFluxes_maxM2wM1noworse_cell = cell(length(M_spec),nMs);
M2rxnFluxes_maxM2wM1noworse_cell = cell(length(M_spec),nMs);
%interactions:
interaction_cell = repmat({sparse(nEnvSamples,nMs)},length(M_spec),1);

tic
for mmSpec = 1:length(M_spec)

    ind1 = M_spec(mmSpec);
    M1 = getHGTmodel_fromFile( ind1, filedr_metmodels );
    %params:
    nrM1=size(M1.S_int,2); %number of reactions in modelA
    bmiM1_smat = M1.bmi;

    %set up temp mat to store interactions:
    temp_interaction_cell = sparse(nEnvSamples,nMs);

    for mmpair = 1:nMs

        if mod(mmpair, 50)==0
            disp(mmpair)
        end

        M2 = getHGTmodel_fromFile( mmpair, filedr_metmodels );
        %params:
        nrM2=size(M2.S_int,2); %number of reactions in modelB\
        bmiM2_smat = nrM1+M2.bmi;

        %now rhs_ub is given by the sum
        env_rhsub = M1.rhs_ext_ub + M2.rhs_ext_ub;
        
        %create combined temp model
        tempmodelM1M2 = createBaseTempmetmodelMiAndMj( M1, M2, nECtot, num_nec, env_rhsub );

        %set up temp mats to store data
        temp_GRs = sparse(nEnvSamples,6);
        temp_ECfluxes_maxM1wM2noworse = sparse(nEnvSamples,nECtot);
        temp_ECfluxes_maxM2wM1noworse = sparse(nEnvSamples,nECtot);
        temp_M1rxnFluxes_maxM1wM2noworse = sparse(nEnvSamples,nrM1);
        temp_M2rxnFluxes_maxM1wM2noworse = sparse(nEnvSamples,nrM2);
        temp_M1rxnFluxes_maxM2wM1noworse = sparse(nEnvSamples,nrM1);
        temp_M2rxnFluxes_maxM2wM1noworse = sparse(nEnvSamples,nrM2);

        for ee = 1:nEnvSamples

            env_rhslb = conc*env_mat(ee,:)';

            lambda_M1nW = indiv_GRs(ee, ind1);
            lambda_M2nW = indiv_GRs(ee, mmpair);
            
            try
                output = JustGrowPairFromTempModels_noWTog_OutputRxnAndECFluxes( lambda_M1nW, lambda_M2nW, tempmodelM1M2, nECtot, env_rhslb, bmiM1_smat, nrM1, bmiM2_smat, nrM2, 0.001  );
                interaction_cat = CategoriseNEW_withStrongComp( output.RatesVec, 0.001, ee);
                temp_interaction_cell(ee,mmpair) = interaction_cat.cat;

                temp_GRs(ee,:) = output.RatesVec;

                if sum(output.RatesVec>0.001)
                    temp_ECfluxes_maxM1wM2noworse(ee,:) = output.ECfluxes_maxM1wM2noworse';
                    temp_ECfluxes_maxM2wM1noworse(ee,:) = output.ECfluxes_maxM2wM1noworse';
    
                    temp_M1rxnFluxes_maxM1wM2noworse(ee,:) = output.M1rxnFluxes_maxM1wM2noworse';
                    temp_M2rxnFluxes_maxM1wM2noworse(ee,:) = output.M2rxnFluxes_maxM1wM2noworse';
                    temp_M1rxnFluxes_maxM2wM1noworse(ee,:) = output.M1rxnFluxes_maxM2wM1noworse';
                    temp_M2rxnFluxes_maxM2wM1noworse(ee,:) = output.M2rxnFluxes_maxM2wM1noworse';
                end

            catch

                temp_interaction_cell(ee,mmpair) = 17;
    
            end

        end

        %%build cells:
        %growth rates:
        GR_cell{mmSpec, mmpair} = temp_GRs;
        %fluxes etc:
        ECfluxes_maxM1wM2noworse_cell{mmSpec, mmpair} = temp_ECfluxes_maxM1wM2noworse;
        ECfluxes_maxM2wM1noworse_cell{mmSpec, mmpair} = temp_ECfluxes_maxM2wM1noworse;
        M1rxnFluxes_maxM1wM2noworse_cell{mmSpec, mmpair} = temp_M1rxnFluxes_maxM1wM2noworse;
        M2rxnFluxes_maxM1wM2noworse_cell{mmSpec, mmpair} = temp_M2rxnFluxes_maxM1wM2noworse;
        M1rxnFluxes_maxM2wM1noworse_cell{mmSpec, mmpair} = temp_M1rxnFluxes_maxM2wM1noworse;
        M2rxnFluxes_maxM2wM1noworse_cell{mmSpec, mmpair} = temp_M2rxnFluxes_maxM2wM1noworse;

    end

        %interactions:
        interaction_cell{mmSpec} = temp_interaction_cell;

end
toc

%save:
savestr = [savestr_start, 'interaction_cell.mat'];
save(savestr, 'interaction_cell' ) 
savestr = [savestr_start, 'GR_cell.mat'];
save(savestr, 'GR_cell' ) 
savestr = [savestr_start, 'ECfluxes_maxM1wM2noworse_cell.mat'];
save(savestr, 'ECfluxes_maxM1wM2noworse_cell' ) 
savestr = [savestr_start, 'ECfluxes_maxM2wM1noworse_cell.mat'];
save(savestr, 'ECfluxes_maxM2wM1noworse_cell' ) 
savestr = [savestr_start, 'M1rxnFluxes_maxM1wM2noworse_cell.mat'];
save(savestr, 'M1rxnFluxes_maxM1wM2noworse_cell' ) 
savestr = [savestr_start, 'M2rxnFluxes_maxM1wM2noworse_cell.mat'];
save(savestr, 'M2rxnFluxes_maxM1wM2noworse_cell' ) 
savestr = [savestr_start, 'M1rxnFluxes_maxM2wM1noworse_cell.mat'];
save(savestr, 'M1rxnFluxes_maxM2wM1noworse_cell' ) 
savestr = [savestr_start, 'M2rxnFluxes_maxM2wM1noworse_cell.mat'];
save(savestr, 'M2rxnFluxes_maxM2wM1noworse_cell' ) 
                

