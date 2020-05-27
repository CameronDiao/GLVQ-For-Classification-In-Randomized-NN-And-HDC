printf('lendo problema %s ...\n', problema);

n_entradas= 64; n_clases= 100; n_fich= 1; fich{1}= 'data_Mar_64.txt'; n_patrons(1)= 1600;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;

clase={'AcerCampestre', 'AcerCapillipes', 'AcerCircinatum', 'AcerMono', 'AcerOpalus', 'AcerPalmatum', 'AcerPictum', 'AcerPlatanoids', 'AcerRubrum', 'AcerRufinerve', 'AcerSaccharinum', 'AlnusCordata', 'AlnusMaximowiczii', 'AlnusRubra', 'AlnusSieboldiana', 'AlnusViridis', 'ArundinariaSimonii', 'BetulaAustrosinensis', 'BetulaPendula', 'CallicarpaBodinieri', 'CastaneaSativa', 'CeltisKoraiensis', 'CercisSiliquastrum', 'CornusChinensis', 'CornusControversa', 'CornusMacrophylla', 'CotinusCoggygria', 'CrataegusMonogyna', 'CytisusBattandieri', 'EucalyptusGlaucescens', 'EucalyptusNeglecta', 'EucalyptusUrnigera', 'FagusSylvatica', 'GinkgoBiloba', 'IlexAquifolium', 'IlexCornuta', 'LiquidambarStyraciflua', 'LiriodendronTulipifera', 'LithocarpusCleistocarpus', 'LithocarpusEdulis', 'MagnoliaHeptapeta', 'MagnoliaSalicifolia', 'MorusNigra', 'OleaEuropaea', 'Phildelphus', 'PopulusAdenopoda', 'PopulusGrandidentata', 'PopulusNigra', 'PrunusAvium', 'PrunusXShmittii', 'PterocaryaStenoptera', 'QuercusAfares', 'QuercusAgrifolia', 'QuercusAlnifolia', 'QuercusBrantii', 'QuercusCanariensis', 'QuercusCastaneifolia', 'QuercusCerris', 'QuercusChrysolepis', 'QuercusCoccifera', 'QuercusCoccinea', 'QuercusCrassifolia', 'QuercusCrassipes', 'QuercusDolicholepis', 'QuercusEllipsoidalis', 'QuercusGreggii', 'QuercusHartwissiana', 'QuercusIlex', 'QuercusImbricaria', 'QuercusInfectoriasub', 'QuercusKewensis', 'QuercusNigra', 'QuercusPalustris', 'QuercusPhellos', 'QuercusPhillyraeoides', 'QuercusPontica', 'QuercusPubescens', 'QuercusPyrenaica', 'QuercusRhysophylla', 'QuercusRubra', 'QuercusSemecarpifolia', 'QuercusShumardii', 'QuercusSuber', 'QuercusTexana', 'QuercusTrojana', 'QuercusVariabilis', 'QuercusVulcanica', 'QuercusxHispanica', 'QuercusxTurneri', 'RhododendronxRussellianum', 'SalixFragilis', 'SalixIntergra', 'SorbusAria', 'TiliaOliveri', 'TiliaPlatyphyllos', 'TiliaTomentosa', 'UlmusBergmanniana', 'ViburnumTinus', 'ViburnumxRhytidophylloides', 'ZelkovaSerrata'};

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	t=fscanf(f,'%s',1);
	for j=1:n_clases
	  if strcmp(t,clase{j})
		cl(i_fich,i)=j-1; break
	  end
	end
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
	end	
  end
  fclose(f);
end
