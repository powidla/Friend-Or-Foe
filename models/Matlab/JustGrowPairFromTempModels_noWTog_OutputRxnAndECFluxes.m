function output = JustGrowPairFromTempModels_noWTog_OutputRxnAndECFluxes( lambda_M1nW, lambda_M2nW, tempmodelM1M2, num_ec, env_rhslb, bmiM1_smat, nrM1, bmiM2_smat, nrM2, nWtol  )

    % optimise M1 such that M2 does no worse:
    tempmodelM1wM2nW = tempmodelM1M2;
    tempmodelM1wM2nW.A(end,bmiM2_smat) = 1;
    tempmodelM1wM2nW.A = sparse(tempmodelM1wM2nW.A);
    tempmodelM1wM2nW.rhs(end) = lambda_M2nW-nWtol;
    tempmodelM1wM2nW.obj(bmiM1_smat)=-1;
    output_M1wM2nW = runTempmodelMiAndMj_nWconstraints_extras( tempmodelM1wM2nW, num_ec, env_rhslb, bmiM2_smat );  
    % optimise M1 such that M2 does no worse:
    tempmodelM2wM1nW = tempmodelM1M2;
    tempmodelM2wM1nW.A(end,bmiM1_smat) = 1;
    tempmodelM2wM1nW.A = sparse(tempmodelM2wM1nW.A);
    tempmodelM2wM1nW.rhs(end) = lambda_M1nW-nWtol;
    tempmodelM2wM1nW.obj(bmiM2_smat) = -1;
    output_M2wM1nW = runTempmodelMiAndMj_nWconstraints_extras( tempmodelM2wM1nW, num_ec, env_rhslb, bmiM1_smat );
    
    output.RatesVec = [ lambda_M1nW, output_M1wM2nW.RateOptMx, output_M2wM1nW.RateNoWorse, lambda_M2nW, output_M1wM2nW.RateNoWorse,output_M2wM1nW.RateOptMx ];   %row vector of the growth rates 
    
    %Rxn fluxes:
    tempM1M2fluxes = output_M1wM2nW.rxnfluxes;
    tempM1fluxes = tempM1M2fluxes(1:nrM1);
    tempM2fluxes = tempM1M2fluxes(nrM1+1:nrM1+nrM2);
    output.M1rxnFluxes_maxM1wM2noworse = tempM1fluxes;
    output.M2rxnFluxes_maxM1wM2noworse = tempM2fluxes;
    tempM1M2fluxes = output_M2wM1nW.rxnfluxes;
    tempM1fluxes = tempM1M2fluxes(1:nrM1);
    tempM2fluxes = tempM1M2fluxes(nrM1+1:nrM1+nrM2);
    output.M1rxnFluxes_maxM2wM1noworse = tempM1fluxes;
    output.M2rxnFluxes_maxM2wM1noworse = tempM2fluxes;


    %EC fluxes:
    output.ECfluxes_maxM1wM2noworse = output_M1wM2nW.ECfluxMat;
    output.ECfluxes_maxM2wM1noworse = output_M2wM1nW.ECfluxMat;

end