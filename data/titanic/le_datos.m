printf('lendo problema %s ...\n', problema);

n_entradas= 3; n_clases= 2; n_fich= 1; fich{1}= 'Dataset.data'; n_patrons(1)= 2201;

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
	  t = fscanf(f,'%s',1);
	  if j==1
		val={'1st', '2nd', '3rd', 'crew'};
	  elseif j==2
		val={'adult','child'};
	  elseif j==3
		val={'female','male'};
	  end
	  n=length(val);
	  for k=1:n
		if strcmp(t,val{k})
		  x(i_fich,i,j)=k; break
		end
	  end
	end	
	t = fscanf(f,'%s',1);  	% lectura da clase
	if strcmp(t, 'no')
	  cl(i_fich,i)= 0;
	elseif strcmp(t, 'yes')
	  cl(i_fich,i)=1;
	else
	  error('clase %s descoñecida', t)
	end
  end
  fclose(f);
end
