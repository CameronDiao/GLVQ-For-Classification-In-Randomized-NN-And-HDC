printf('lendo problema %s ...\n', problema);

n_entradas= 50; n_clases= 2; n_fich= 1; fich{1}= 'MiniBooNE_PID.txt'; n_patrons(1)= 130064;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  n1 = fscanf(f,'%i', 1); n2 = fscanf(f,'%i', 1);
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%e',1);
	end	
	if i <= n1
	  cl(i_fich,i) = 0;   % clase: neutrinos (sinal)
	else
	  cl(i_fich,i) = 1;   % clase neutrinos muón (fondo)
	end
  end
  fclose(f);
end
