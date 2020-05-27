printf('lendo problema %s ...\n', problema);

n_entradas= 9; n_clases= 2; n_fich= 1; fich{1}= 'breast-cancer-wisconsin.data'; n_patrons(1)= 699; %fich{2}= ' '; n_patrons(2)= 0;

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
	fscanf(f,'%i',1); fscanf(f,'%c',1);   % le e descarta o ID e a coma
	for j = 1:n_entradas
	  t = fscanf(f,'%c',1);
	  if t ~= '?'
		fseek(f,-1,SEEK_CUR); t = fscanf(f, '%i',1); x(i_fich,i,j) = t;
	  else
		x(i_fich,i,j) = 0;
	  end
	  fscanf(f,'%c',1);  % le e descarta a coma
	end	
	t = fscanf(f,'%i',1);  % lectura da clase
	if t==2
	  cl(i_fich,i) = 0;  % benigno
	else
	  cl(i_fich,i) = 1;  % maligno
	end
  end
  fclose(f);
end
