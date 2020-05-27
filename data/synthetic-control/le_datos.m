printf('lendo problema %s ...\n', problema);

n_entradas= 60; n_clases= 6; n_fich= 1; fich{1}= 'synthetic_control.data'; n_patrons(1)= 600;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
	end	
	if i<=100
	  cl(i_fich,i) = 0; % normal
	elseif i<= 200
	  cl(i_fich,i) = 1; % cyclic
	elseif i<= 300
	  cl(i_fich,i) = 2; % Increasing trend
	elseif i<= 400
	  cl(i_fich,i) = 3; % Decreasing trend
	elseif i<= 500
	  cl(i_fich,i) = 4; % Upward shift
	else
	  cl(i_fich,i) = 5; % Downward shift
	end
  end
  fclose(f);
end
