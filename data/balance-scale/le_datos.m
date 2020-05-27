printf('lendo problema %s ...\n', problema);

n_entradas= 4; n_clases= 3; n_fich= 1; fich{1}= 'balance-scale.data'; n_patrons(1)= 625; %fich{2}= ' '; n_patrons(2)= 0;

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
	t = fscanf(f,'%c',1); fscanf(f,'%c',1);  %le a clase
	if t=='B'
	  cl(i_fich,i)= 0;
	elseif t=='L'
	  cl(i_fich,i)= 1;
	elseif t=='R'
	  cl(i_fich,i)= 2;
	end
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%i',1); fscanf(f,'%c',1); 
	end
  end
  fclose(f);
end
