% annealing
printf('lendo problema %s ...\n', problema);

n_entradas= 279; n_clases= 13; % ollo: hai 16 clases, pero as clases 11, 12 e 13 non teñen patróns
n_fich= 1; fich{1}= 'arrhythmia.data'; n_patrons(1)= 452; %fich{2}= ' '; n_patrons(2)= 0;

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
	  t = fscanf(f,'%c',1);
	  if t ~= '?'
		fseek(f,-1,SEEK_CUR); t = fscanf(f, '%f',1); x(i_fich,i,j) = t;
	  else
		x(i_fich,i,j) = 0;
	  end
	  fscanf(f,'%c',1);  % le e descarta a coma
	end	
	t= fscanf(f,'%i',1);  	% lectura da clase
	if t>=14
	  t=t-3;
	end
	cl(i_fich,i) = t - 1;
  end
  fclose(f);
end
