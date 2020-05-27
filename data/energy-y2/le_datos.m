printf('lendo problema %s ...\n', problema);

n_entradas= 8; n_clases= 3; n_fich= 1; fich{1}= 'ENB2012_data.csv'; n_patrons(1)= 768;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  fscanf(f,'%s',n_entradas+2);
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
	end
	fscanf(f,'%g',1);  % le e descarta Y1
	cl(i_fich,i)= fscanf(f,'%g',1);  % lectura de clase Y2
  end
  fclose(f);
end


vmin=min(cl(i_fich,:)); vmax=max(cl(i_fich,:)); rango=vmax - vmin;
cl(i_fich,:) = (cl(i_fich,:) - vmin)/rango;
for i=1:n_patrons(1)
  if cl(i_fich,i) < 0.3
	cl(i_fich,i)=0;
  elseif cl(i_fich,i) < 0.6
	cl(i_fich,i)=1;
  else
	cl(i_fich,i)=2;
  end
end
