function Microbe = getHGTmodel_fromFile( ind, filedr )
%function takes a microbe index and the collection as input and outputs the
%metabolic model
%ind should be input as a number
%collection should be input in ''
%filedr should be input in ''

load([filedr, 'model',num2str(ind),'.mat']);
Microbe = MetabModel;
clear MetabModel

end