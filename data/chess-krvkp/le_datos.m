printf('lendo problema %s ...\n', problema);

n_entradas= 36; n_clases= 2; n_fich= 1; fich{1}= 'kr-vs-kp.data'; n_patrons(1)= 3196;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
clase={'nowin', 'won'};
for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if j==13
		val={'l','g'};
	  elseif j==15
		val={'b','n','w'};
	  elseif j==36
		val={'n','t'};
	  else
		val={'f','t'};
	  end
	  n=length(val);
	  for k=1:n
		if strcmp(t,val{k})
		  x(i_fich,i,j)=k; break
		end
	  end
	end	
	t = fscanf(f,'%s',1);
	for j=1:n_clases    	% lectura da clase
	  if strcmp(t,clase{j})
		cl(i_fich,i)=j-1; break
	  end
	end
  end
  fclose(f);
end
